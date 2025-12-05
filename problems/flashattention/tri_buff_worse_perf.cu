#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <float.h>

// --- Core Architecture & Problem Parameters ---
constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;

// --- Kernel Tuning Parameters ---
constexpr int BLOCK_ROWS_Q = 128;
constexpr int BLOCK_COLS_KV = 32;
constexpr int NUM_WARPS = 4;
constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE;

// --- OPTIMIZED: Use 3-buffer instead of 4 (better SM occupancy) ---
constexpr int NUM_BUFFERS = 3;

// --- Tensor Core MMA Instruction Shape ---
constexpr int MMA_ROWS_M = 16;
constexpr int MMA_COLS_N = 8;
constexpr int MMA_DIM_K = 16;

__device__ __host__ constexpr int div_up(int a, int b) { 
    return (a + b - 1) / b; 
}

template <int STRIDE_BYTES>
__device__ uint32_t swizzle_smem_addr(uint32_t addr) {
    if constexpr (STRIDE_BYTES == 16) {
        return addr;
    }
    uint32_t row_in_tile = (addr / STRIDE_BYTES) % 8;
    uint32_t bits_to_xor = row_in_tile / max(64 / STRIDE_BYTES, 1);
    return addr ^ (bits_to_xor << 4);
}

template <int TILE_HEIGHT, int TILE_WIDTH>
__device__ inline void copy_gmem_to_smem_swizzled(
    uint32_t smem_base_addr, 
    const __half *gmem_ptr, 
    int gmem_stride, 
    int thread_idx
) {
    constexpr int ELEMS_PER_COPY = 16 / sizeof(__half);
    constexpr int ITERS = TILE_HEIGHT * TILE_WIDTH / (THREADS_PER_BLOCK * ELEMS_PER_COPY);

    for (int i = 0; i < ITERS; i++) {
        const int elem_idx = (i * THREADS_PER_BLOCK + thread_idx) * ELEMS_PER_COPY;
        const int row = elem_idx / TILE_WIDTH;
        const int col = elem_idx % TILE_WIDTH;

        if (row < TILE_HEIGHT && col + ELEMS_PER_COPY <= TILE_WIDTH) {
            const uint32_t dst_addr = swizzle_smem_addr<TILE_WIDTH * sizeof(__half)>(
                smem_base_addr + (row * TILE_WIDTH + col) * sizeof(__half));
            const __half *src_addr = gmem_ptr + row * gmem_stride + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
        }
    }
}

// --- PTX Assembly Wrappers ---
__device__ inline void load_matrix_8x16_f16(uint32_t regs[4], uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(smem_addr));
}

__device__ inline void load_matrix_8x8_f16(uint32_t regs[2], uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1]) : "r"(smem_addr));
}

__device__ inline void load_matrix_8x8_f16_transposed(uint32_t regs[2], uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1]) : "r"(smem_addr));
}

__device__ inline void mma_16x8x16_f32_f16(uint32_t a[4], uint32_t b[2], float d[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
              : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
              : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK) flash_attention_optimized_pipeline_kernel(
    const __half* __restrict__ gmem_q,
    const __half* __restrict__ gmem_k,
    const __half* __restrict__ gmem_v,
    __half* __restrict__ gmem_o,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float softmax_scale) {

    // --- Block & Thread Identification ---
    const int block_idx_x = blockIdx.x;
    const int block_idx_y = blockIdx.y;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;

    // --- Work Scheduling ---
    const int q_blocks_per_head = div_up(seq_len, BLOCK_ROWS_Q);
    const int head_idx = block_idx_y % num_heads;
    const int batch_idx = block_idx_y / num_heads;
    const int q_block_idx = block_idx_x % q_blocks_per_head;
    
    const int q_block_row_start = q_block_idx * BLOCK_ROWS_Q;

    // --- Global Memory Pointers ---
    const int head_gmem_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const __half* query_gmem_ptr = gmem_q + head_gmem_offset + q_block_row_start * HEAD_DIM;
    const __half* key_gmem_base_ptr = gmem_k + head_gmem_offset;
    const __half* value_gmem_base_ptr = gmem_v + head_gmem_offset;
    __half* output_gmem_ptr = gmem_o + head_gmem_offset + q_block_row_start * HEAD_DIM;

    // --- Shared Memory Layout (TRIPLE-BUFFERED for K and V) ---
    extern __shared__ __half shared_mem[];
    const uint32_t smem_q_base = __cvta_generic_to_shared(shared_mem);
    const uint32_t smem_k_base = smem_q_base; // K reuses Q's space
    const uint32_t smem_v_base = smem_k_base + NUM_BUFFERS * BLOCK_COLS_KV * HEAD_DIM * sizeof(__half);

    // --- Register File Allocation ---
    constexpr int Q_ROWS_PER_WARP = BLOCK_ROWS_Q / NUM_WARPS;
    constexpr int Q_TILES_PER_WARP = Q_ROWS_PER_WARP / MMA_ROWS_M;
    constexpr int KV_TILES_PER_BLOCK = BLOCK_COLS_KV / MMA_COLS_N;
    constexpr int DIM_TILES = HEAD_DIM / MMA_DIM_K;

    uint32_t regs_q[Q_TILES_PER_WARP][DIM_TILES][4];
    uint32_t regs_k[KV_TILES_PER_BLOCK][DIM_TILES][2];
    uint32_t regs_p[Q_TILES_PER_WARP][BLOCK_COLS_KV / MMA_DIM_K][4];
    uint32_t regs_v[BLOCK_COLS_KV / MMA_DIM_K][HEAD_DIM / MMA_COLS_N][2];
    float regs_o[Q_TILES_PER_WARP][HEAD_DIM / MMA_COLS_N][4] = {};

    // --- Pre-compute swizzled addresses ---
    uint32_t thread_smem_q_addr, thread_smem_k_addr, thread_smem_v_addr;
    {
        const int row = warp_idx * Q_ROWS_PER_WARP + (lane_idx % 16);
        const int col = (lane_idx / 16) * 8;
        thread_smem_q_addr = swizzle_smem_addr<HEAD_DIM * sizeof(__half)>(
            smem_q_base + (row * HEAD_DIM + col) * sizeof(__half));
    }
    {
        const int row = lane_idx % 8;
        const int col = (lane_idx / 8) * 8;
        thread_smem_k_addr = swizzle_smem_addr<HEAD_DIM * sizeof(__half)>(
            smem_k_base + (row * HEAD_DIM + col) * sizeof(__half));
    }
    {
        const int row = lane_idx % 16;
        const int col = (lane_idx / 16) * 8;
        thread_smem_v_addr = swizzle_smem_addr<HEAD_DIM * sizeof(__half)>(
            smem_v_base + (row * HEAD_DIM + col) * sizeof(__half));
    }

    // --- Online Softmax State ---
    float running_row_max[Q_TILES_PER_WARP][2];
    float running_row_sum_exp[Q_TILES_PER_WARP][2] = {};
    for (int i = 0; i < Q_TILES_PER_WARP; ++i) {
        running_row_max[i][0] = -FLT_MAX;
        running_row_max[i][1] = -FLT_MAX;
    }

    // --- Load Q tile ---
    copy_gmem_to_smem_swizzled<BLOCK_ROWS_Q, HEAD_DIM>(smem_q_base, query_gmem_ptr, HEAD_DIM, thread_idx);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Load Q from shared memory into registers
    for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
        for (int dim_tile_idx = 0; dim_tile_idx < DIM_TILES; ++dim_tile_idx) {
            uint32_t addr = thread_smem_q_addr + q_tile_idx * MMA_ROWS_M * HEAD_DIM * sizeof(__half);
            addr ^= dim_tile_idx * MMA_DIM_K * sizeof(__half);
            load_matrix_8x16_f16(regs_q[q_tile_idx][dim_tile_idx], addr);
        }
    }
    __syncthreads();

    // --- SOFTWARE PIPELINE OPTIMIZATION ---
    const int kv_iters = div_up(seq_len, BLOCK_COLS_KV);
    
    // Helper: Load both K and V for an iteration
    auto load_kv_tile = [&](int iter) {
        if (iter >= kv_iters) return;
        const int buffer_idx = iter % NUM_BUFFERS;
        
        // Load K
        const uint32_t k_dst_addr = smem_k_base + buffer_idx * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
        const __half* k_src_ptr = key_gmem_base_ptr + iter * BLOCK_COLS_KV * HEAD_DIM;
        copy_gmem_to_smem_swizzled<BLOCK_COLS_KV, HEAD_DIM>(k_dst_addr, k_src_ptr, HEAD_DIM, thread_idx);
        asm volatile("cp.async.commit_group;");
        
        // Load V
        const uint32_t v_dst_addr = smem_v_base + buffer_idx * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
        const __half* v_src_ptr = value_gmem_base_ptr + iter * BLOCK_COLS_KV * HEAD_DIM;
        copy_gmem_to_smem_swizzled<BLOCK_COLS_KV, HEAD_DIM>(v_dst_addr, v_src_ptr, HEAD_DIM, thread_idx);
        asm volatile("cp.async.commit_group;");
    };

    // ========================================================================
    // PROLOGUE: Prefetch first 2 iterations
    // ========================================================================
    load_kv_tile(0);
    load_kv_tile(1);

    // ========================================================================
    // MAIN LOOP: Optimized 3-stage pipeline
    // ========================================================================
    for (int kv_iter = 0; kv_iter < kv_iters; ++kv_iter) {
        const int buffer_idx = kv_iter % NUM_BUFFERS;
        float regs_s[Q_TILES_PER_WARP][KV_TILES_PER_BLOCK][4] = {};

        // --- Prefetch next iteration (kv_iter + 2) ---
        load_kv_tile(kv_iter + 2);

        // --- Wait for current K to be ready ---
        // With 3 buffers, we have 2*3 = 6 groups max in flight
        // Wait until only 3 groups remain (allows current K to be ready)
        asm volatile("cp.async.wait_group 3;");
        __syncthreads();

        // --- Load K from shared to registers ---
        for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < DIM_TILES; ++dim_tile_idx) {
                uint32_t addr = thread_smem_k_addr + buffer_idx * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
                addr += kv_tile_idx * MMA_COLS_N * HEAD_DIM * sizeof(__half);
                addr ^= dim_tile_idx * MMA_DIM_K * sizeof(__half);
                load_matrix_8x8_f16(regs_k[kv_tile_idx][dim_tile_idx], addr);
            }
        }
        
        // --- Compute S = Q @ K^T ---
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                for (int dim_tile_idx = 0; dim_tile_idx < DIM_TILES; ++dim_tile_idx) {
                    mma_16x8x16_f32_f16(regs_q[q_tile_idx][dim_tile_idx], 
                                       regs_k[kv_tile_idx][dim_tile_idx], 
                                       regs_s[q_tile_idx][kv_tile_idx]);
                }
            }
        }

        // --- Online Softmax: Compute P = softmax(S) ---
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            // Find row max
            float local_row_max[2] = {-FLT_MAX, -FLT_MAX};
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                float *s_vals = regs_s[q_tile_idx][kv_tile_idx];
                for(int i = 0; i < 4; ++i) s_vals[i] *= softmax_scale;
                local_row_max[0] = fmaxf(local_row_max[0], fmaxf(s_vals[0], s_vals[1]));
                local_row_max[1] = fmaxf(local_row_max[1], fmaxf(s_vals[2], s_vals[3]));
            }
            
            // Warp reduce for row max
            for (int offset = 1; offset < 4; offset *= 2) {
                local_row_max[0] = fmaxf(local_row_max[0], __shfl_xor_sync(0xFFFFFFFF, local_row_max[0], offset));
                local_row_max[1] = fmaxf(local_row_max[1], __shfl_xor_sync(0xFFFFFFFF, local_row_max[1], offset));
            }
            
            // Update running max and compute rescale factor
            float new_row_max[2] = {
                fmaxf(local_row_max[0], running_row_max[q_tile_idx][0]), 
                fmaxf(local_row_max[1], running_row_max[q_tile_idx][1])
            };
            float rescale[2] = {
                __expf(running_row_max[q_tile_idx][0] - new_row_max[0]), 
                __expf(running_row_max[q_tile_idx][1] - new_row_max[1])
            };
            
            // Rescale previous output
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                regs_o[q_tile_idx][dim_tile_idx][0] *= rescale[0]; 
                regs_o[q_tile_idx][dim_tile_idx][1] *= rescale[0];
                regs_o[q_tile_idx][dim_tile_idx][2] *= rescale[1]; 
                regs_o[q_tile_idx][dim_tile_idx][3] *= rescale[1];
            }
            running_row_max[q_tile_idx][0] = new_row_max[0];
            running_row_max[q_tile_idx][1] = new_row_max[1];
            
            // Compute exp and sum
            float local_row_sum_exp[2] = {};
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                float *s_vals = regs_s[q_tile_idx][kv_tile_idx];
                s_vals[0] = __expf(s_vals[0] - new_row_max[0]); 
                s_vals[1] = __expf(s_vals[1] - new_row_max[0]);
                s_vals[2] = __expf(s_vals[2] - new_row_max[1]); 
                s_vals[3] = __expf(s_vals[3] - new_row_max[1]);
                local_row_sum_exp[0] += s_vals[0] + s_vals[1]; 
                local_row_sum_exp[1] += s_vals[2] + s_vals[3];
                
                // Convert to half precision and store in P registers
                __half2* p_vals_ptr = reinterpret_cast<__half2*>(regs_p[q_tile_idx][kv_tile_idx / 2]);
                p_vals_ptr[(kv_tile_idx % 2) * 2]     = __float22half2_rn(make_float2(s_vals[0], s_vals[1]));
                p_vals_ptr[(kv_tile_idx % 2) * 2 + 1] = __float22half2_rn(make_float2(s_vals[2], s_vals[3]));
            }
            
            // Warp reduce for sum
            for (int offset = 1; offset < 4; offset *= 2) {
                local_row_sum_exp[0] += __shfl_xor_sync(0xFFFFFFFF, local_row_sum_exp[0], offset);
                local_row_sum_exp[1] += __shfl_xor_sync(0xFFFFFFFF, local_row_sum_exp[1], offset);
            }
            
            // Update running sum
            running_row_sum_exp[q_tile_idx][0] = running_row_sum_exp[q_tile_idx][0] * rescale[0] + local_row_sum_exp[0];
            running_row_sum_exp[q_tile_idx][1] = running_row_sum_exp[q_tile_idx][1] * rescale[1] + local_row_sum_exp[1];
        }

        // --- Wait for V to be ready ---
        // Wait until only 2 groups remain (current V is ready)
        asm volatile("cp.async.wait_group 2;");
        __syncthreads();

        // --- Load V from shared to registers ---
        for (int kv_tile_idx = 0; kv_tile_idx < (BLOCK_COLS_KV / MMA_DIM_K); ++kv_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                uint32_t addr = thread_smem_v_addr + buffer_idx * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
                addr += kv_tile_idx * MMA_DIM_K * HEAD_DIM * sizeof(__half);
                addr ^= dim_tile_idx * MMA_COLS_N * sizeof(__half);
                load_matrix_8x8_f16_transposed(regs_v[kv_tile_idx][dim_tile_idx], addr);
            }
        }
        
        // --- Compute O += P @ V ---
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                for (int kv_tile_idx = 0; kv_tile_idx < (BLOCK_COLS_KV / MMA_DIM_K); ++kv_tile_idx) {
                    mma_16x8x16_f32_f16(regs_p[q_tile_idx][kv_tile_idx], 
                                       regs_v[kv_tile_idx][dim_tile_idx], 
                                       regs_o[q_tile_idx][dim_tile_idx]);
                }
            }
        }
        
        __syncthreads();  // Ensure all warps finish before buffer reuse
    }

    // ========================================================================
    // Finalize and Write Output
    // ========================================================================
    for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
        for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
            const int row = warp_idx * Q_ROWS_PER_WARP + q_tile_idx * MMA_ROWS_M + (lane_idx / 4);
            const int col = dim_tile_idx * MMA_COLS_N + (lane_idx % 4) * 2;
            
            if (q_block_row_start + row >= seq_len) continue;

            float* o_vals = regs_o[q_tile_idx][dim_tile_idx];
            o_vals[0] /= running_row_sum_exp[q_tile_idx][0]; 
            o_vals[1] /= running_row_sum_exp[q_tile_idx][0];
            o_vals[2] /= running_row_sum_exp[q_tile_idx][1]; 
            o_vals[3] /= running_row_sum_exp[q_tile_idx][1];

            *reinterpret_cast<__half2*>(output_gmem_ptr + (row + 0) * HEAD_DIM + col) = 
                __float22half2_rn(make_float2(o_vals[0], o_vals[1]));
            *reinterpret_cast<__half2*>(output_gmem_ptr + (row + 8) * HEAD_DIM + col) = 
                __float22half2_rn(make_float2(o_vals[2], o_vals[3]));
        }
    }
}

// --- PyTorch Wrapper ---
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = torch::empty_like(Q);
    const int batch_size = Q.size(0), num_heads = Q.size(1), seq_len = Q.size(2), head_dim = Q.size(3);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    TORCH_CHECK(head_dim == 128, "This kernel only supports head_dim=128");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    
    // Shared memory size for TRIPLE-BUFFERING K and V (3 buffers each)
    const int smem_size = max(BLOCK_ROWS_Q, NUM_BUFFERS * BLOCK_COLS_KV * 2) * HEAD_DIM * sizeof(__half);
    if (smem_size > 48000) {
        cudaFuncSetAttribute(flash_attention_optimized_pipeline_kernel, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 grid(div_up(seq_len, BLOCK_ROWS_Q), batch_size * num_heads);
    dim3 block(THREADS_PER_BLOCK);
  
    flash_attention_optimized_pipeline_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), 
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        batch_size, num_heads, seq_len, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(err));

    return O;
}

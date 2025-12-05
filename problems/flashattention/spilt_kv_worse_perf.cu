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

// --- Split-K Parameters ---
constexpr int SPLIT_K = 4;  // Split KV sequence into 4 chunks

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

// ============================================================================
// SPLIT-K KERNEL: Each block processes a chunk of the KV sequence
// ============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK) 
flash_attention_splitk_kernel(
    const __half* __restrict__ gmem_q,
    const __half* __restrict__ gmem_k,
    const __half* __restrict__ gmem_v,
    float* __restrict__ partial_o,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float softmax_scale) {  // Remove split_idx parameter

    // Get split_idx from blockIdx.z
    const int split_idx = blockIdx.z;

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

    // --- Split-K: Determine KV range for this split ---
    const int kv_seq_len_per_split = div_up(seq_len, SPLIT_K);
    const int kv_start = split_idx * kv_seq_len_per_split;
    const int kv_end = min(kv_start + kv_seq_len_per_split, seq_len);
    const int kv_len = kv_end - kv_start;
    
    if (kv_len <= 0) return;  // Nothing to process

    // --- Global Memory Pointers ---
    const int head_gmem_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const __half* query_gmem_ptr = gmem_q + head_gmem_offset + q_block_row_start * HEAD_DIM;
    const __half* key_gmem_base_ptr = gmem_k + head_gmem_offset + kv_start * HEAD_DIM;
    const __half* value_gmem_base_ptr = gmem_v + head_gmem_offset + kv_start * HEAD_DIM;

    // --- Shared Memory Layout ---
    extern __shared__ __half shared_mem[];
    const uint32_t smem_q_base = __cvta_generic_to_shared(shared_mem);
    const uint32_t smem_k_base = smem_q_base;
    const uint32_t smem_v_base = smem_k_base + 2 * BLOCK_COLS_KV * HEAD_DIM * sizeof(__half);

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

    // --- Pipelining Setup ---
    const int kv_iters = div_up(kv_len, BLOCK_COLS_KV);
    
    // Lambdas for starting async copies
    auto load_k_tile = [&](int iter) {
        if (iter < kv_iters) {
            const uint32_t dst_addr = smem_k_base + (iter % 2) * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
            const __half* src_ptr = key_gmem_base_ptr + iter * BLOCK_COLS_KV * HEAD_DIM;
            copy_gmem_to_smem_swizzled<BLOCK_COLS_KV, HEAD_DIM>(dst_addr, src_ptr, HEAD_DIM, thread_idx);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto load_v_tile = [&](int iter) {
        if (iter < kv_iters) {
            const uint32_t dst_addr = smem_v_base + (iter % 2) * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
            const __half* src_ptr = value_gmem_base_ptr + iter * BLOCK_COLS_KV * HEAD_DIM;
            copy_gmem_to_smem_swizzled<BLOCK_COLS_KV, HEAD_DIM>(dst_addr, src_ptr, HEAD_DIM, thread_idx);
        }
        asm volatile("cp.async.commit_group;");
    };

    // --- Prefetch first K and V tiles ---
    load_k_tile(0);
    load_v_tile(0);

    // --- Main Pipelined Loop (over partial KV sequence) ---
    for (int kv_iter = 0; kv_iter < kv_iters; ++kv_iter) {
        float regs_s[Q_TILES_PER_WARP][KV_TILES_PER_BLOCK][4] = {};

        // Wait for K
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // Prefetch next K
        if (kv_iter + 1 < kv_iters) {
            load_k_tile(kv_iter + 1);
        }

        // Load K from shared to registers
        for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < DIM_TILES; ++dim_tile_idx) {
                uint32_t addr = thread_smem_k_addr + (kv_iter % 2) * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
                addr += kv_tile_idx * MMA_COLS_N * HEAD_DIM * sizeof(__half);
                addr ^= dim_tile_idx * MMA_DIM_K * sizeof(__half);
                load_matrix_8x8_f16(regs_k[kv_tile_idx][dim_tile_idx], addr);
            }
        }
        
        // Compute S = Q @ K^T
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                for (int dim_tile_idx = 0; dim_tile_idx < DIM_TILES; ++dim_tile_idx) {
                    mma_16x8x16_f32_f16(regs_q[q_tile_idx][dim_tile_idx], regs_k[kv_tile_idx][dim_tile_idx], regs_s[q_tile_idx][kv_tile_idx]);
                }
            }
        }

        // Online Softmax
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            float local_row_max[2] = {-FLT_MAX, -FLT_MAX};
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                float *s_vals = regs_s[q_tile_idx][kv_tile_idx];
                for(int i=0; i<4; ++i) s_vals[i] *= softmax_scale;
                local_row_max[0] = fmaxf(local_row_max[0], fmaxf(s_vals[0], s_vals[1]));
                local_row_max[1] = fmaxf(local_row_max[1], fmaxf(s_vals[2], s_vals[3]));
            }
            for (int offset = 1; offset < 4; offset *= 2) {
                local_row_max[0] = fmaxf(local_row_max[0], __shfl_xor_sync(0xFFFFFFFF, local_row_max[0], offset));
                local_row_max[1] = fmaxf(local_row_max[1], __shfl_xor_sync(0xFFFFFFFF, local_row_max[1], offset));
            }
            float new_row_max[2] = {fmaxf(local_row_max[0], running_row_max[q_tile_idx][0]), fmaxf(local_row_max[1], running_row_max[q_tile_idx][1])};
            float rescale[2] = {__expf(running_row_max[q_tile_idx][0] - new_row_max[0]), __expf(running_row_max[q_tile_idx][1] - new_row_max[1])};
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                regs_o[q_tile_idx][dim_tile_idx][0] *= rescale[0]; regs_o[q_tile_idx][dim_tile_idx][1] *= rescale[0];
                regs_o[q_tile_idx][dim_tile_idx][2] *= rescale[1]; regs_o[q_tile_idx][dim_tile_idx][3] *= rescale[1];
            }
            running_row_max[q_tile_idx][0] = new_row_max[0];
            running_row_max[q_tile_idx][1] = new_row_max[1];
            float local_row_sum_exp[2] = {};
            for (int kv_tile_idx = 0; kv_tile_idx < KV_TILES_PER_BLOCK; ++kv_tile_idx) {
                float *s_vals = regs_s[q_tile_idx][kv_tile_idx];
                s_vals[0] = __expf(s_vals[0] - new_row_max[0]); s_vals[1] = __expf(s_vals[1] - new_row_max[0]);
                s_vals[2] = __expf(s_vals[2] - new_row_max[1]); s_vals[3] = __expf(s_vals[3] - new_row_max[1]);
                local_row_sum_exp[0] += s_vals[0] + s_vals[1]; local_row_sum_exp[1] += s_vals[2] + s_vals[3];
                __half2* p_vals_ptr = reinterpret_cast<__half2*>(regs_p[q_tile_idx][kv_tile_idx / 2]);
                p_vals_ptr[(kv_tile_idx % 2) * 2]     = __float22half2_rn(make_float2(s_vals[0], s_vals[1]));
                p_vals_ptr[(kv_tile_idx % 2) * 2 + 1] = __float22half2_rn(make_float2(s_vals[2], s_vals[3]));
            }
            for (int offset = 1; offset < 4; offset *= 2) {
                local_row_sum_exp[0] += __shfl_xor_sync(0xFFFFFFFF, local_row_sum_exp[0], offset);
                local_row_sum_exp[1] += __shfl_xor_sync(0xFFFFFFFF, local_row_sum_exp[1], offset);
            }
            running_row_sum_exp[q_tile_idx][0] = running_row_sum_exp[q_tile_idx][0] * rescale[0] + local_row_sum_exp[0];
            running_row_sum_exp[q_tile_idx][1] = running_row_sum_exp[q_tile_idx][1] * rescale[1] + local_row_sum_exp[1];
        }

        // Wait for V
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Prefetch next V
        if (kv_iter + 1 < kv_iters) {
            load_v_tile(kv_iter + 1);
        }

        // Load V from shared to registers
        for (int kv_tile_idx = 0; kv_tile_idx < (BLOCK_COLS_KV / MMA_DIM_K); ++kv_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                uint32_t addr = thread_smem_v_addr + (kv_iter % 2) * (BLOCK_COLS_KV * HEAD_DIM * sizeof(__half));
                addr += kv_tile_idx * MMA_DIM_K * HEAD_DIM * sizeof(__half);
                addr ^= dim_tile_idx * MMA_COLS_N * sizeof(__half);
                load_matrix_8x8_f16_transposed(regs_v[kv_tile_idx][dim_tile_idx], addr);
            }
        }
        
        // Compute O += P @ V
        for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
            for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
                for (int kv_tile_idx = 0; kv_tile_idx < (BLOCK_COLS_KV / MMA_DIM_K); ++kv_tile_idx) {
                    mma_16x8x16_f32_f16(regs_p[q_tile_idx][kv_tile_idx], regs_v[kv_tile_idx][dim_tile_idx], regs_o[q_tile_idx][dim_tile_idx]);
                }
            }
        }
    }

    // ========================================================================
    // Write Partial Results to Global Memory
    // ========================================================================
    // Each split writes its partial output, max, and sum
    // These will be combined in the reduction kernel
    
    const int partial_output_offset = (((batch_idx * num_heads + head_idx) * q_blocks_per_head + q_block_idx) * SPLIT_K + split_idx) * BLOCK_ROWS_Q * HEAD_DIM;
    const int partial_stats_offset = (((batch_idx * num_heads + head_idx) * q_blocks_per_head + q_block_idx) * SPLIT_K + split_idx) * BLOCK_ROWS_Q;
    
    float* partial_o_ptr = partial_o + partial_output_offset;
    float* partial_max_ptr = partial_max + partial_stats_offset;
    float* partial_sum_ptr = partial_sum + partial_stats_offset;

    for (int q_tile_idx = 0; q_tile_idx < Q_TILES_PER_WARP; ++q_tile_idx) {
        for (int dim_tile_idx = 0; dim_tile_idx < (HEAD_DIM / MMA_COLS_N); ++dim_tile_idx) {
            const int row = warp_idx * Q_ROWS_PER_WARP + q_tile_idx * MMA_ROWS_M + (lane_idx / 4);
            const int col = dim_tile_idx * MMA_COLS_N + (lane_idx % 4) * 2;
            
            if (q_block_row_start + row >= seq_len) continue;

            float* o_vals = regs_o[q_tile_idx][dim_tile_idx];
            
            // Write partial output (not normalized yet)
            partial_o_ptr[(row + 0) * HEAD_DIM + col + 0] = o_vals[0];
            partial_o_ptr[(row + 0) * HEAD_DIM + col + 1] = o_vals[1];
            partial_o_ptr[(row + 8) * HEAD_DIM + col + 0] = o_vals[2];
            partial_o_ptr[(row + 8) * HEAD_DIM + col + 1] = o_vals[3];
        }
        
        // Write partial max and sum (only once per row, by one thread)
        if (lane_idx % 4 == 0) {
            const int row = warp_idx * Q_ROWS_PER_WARP + q_tile_idx * MMA_ROWS_M + (lane_idx / 4);
            if (q_block_row_start + row < seq_len) {
                partial_max_ptr[row + 0] = running_row_max[q_tile_idx][0];
                partial_max_ptr[row + 8] = running_row_max[q_tile_idx][1];
                partial_sum_ptr[row + 0] = running_row_sum_exp[q_tile_idx][0];
                partial_sum_ptr[row + 8] = running_row_sum_exp[q_tile_idx][1];
            }
        }
    }
}

// ============================================================================
// REDUCTION KERNEL: Combine partial results from all splits
// ============================================================================
__global__ void flash_attention_reduce_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __half* __restrict__ gmem_o,
    const int batch_size,
    const int num_heads,
    const int seq_len) {
    
    const int q_blocks_per_head = div_up(seq_len, BLOCK_ROWS_Q);
    
    // Each thread processes one output element
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * q_blocks_per_head * BLOCK_ROWS_Q * HEAD_DIM;
    
    if (global_idx >= total_elements) return;
    
    // Decode global index
    const int dim_idx = global_idx % HEAD_DIM;
    const int row_in_block = (global_idx / HEAD_DIM) % BLOCK_ROWS_Q;
    const int q_block_idx = (global_idx / (HEAD_DIM * BLOCK_ROWS_Q)) % q_blocks_per_head;
    const int head_idx = (global_idx / (HEAD_DIM * BLOCK_ROWS_Q * q_blocks_per_head)) % num_heads;
    const int batch_idx = global_idx / (HEAD_DIM * BLOCK_ROWS_Q * q_blocks_per_head * num_heads);
    
    const int row_global = q_block_idx * BLOCK_ROWS_Q + row_in_block;
    if (row_global >= seq_len) return;
    
    // Base offset for this (batch, head, q_block)
    const int base_idx = ((batch_idx * num_heads + head_idx) * q_blocks_per_head + q_block_idx) * SPLIT_K;
    
    // Step 1: Find global max across all splits
    float global_max = -FLT_MAX;
    for (int split_idx = 0; split_idx < SPLIT_K; ++split_idx) {
        const int stats_offset = (base_idx + split_idx) * BLOCK_ROWS_Q + row_in_block;
        global_max = fmaxf(global_max, partial_max[stats_offset]);
    }
    
    // Step 2: Compute rescaled outputs and new sum
    float accumulated_output = 0.0f;
    float global_sum = 0.0f;
    
    for (int split_idx = 0; split_idx < SPLIT_K; ++split_idx) {
        const int output_offset = (base_idx + split_idx) * BLOCK_ROWS_Q * HEAD_DIM + row_in_block * HEAD_DIM + dim_idx;
        const int stats_offset = (base_idx + split_idx) * BLOCK_ROWS_Q + row_in_block;
        
        float local_max = partial_max[stats_offset];
        float local_sum = partial_sum[stats_offset];
        float local_output = partial_o[output_offset];
        
        // Rescale this split's contribution
        float rescale = __expf(local_max - global_max);
        accumulated_output += local_output * rescale;
        global_sum += local_sum * rescale;
    }
    
    // Step 3: Normalize and write final output
    float final_output = accumulated_output / global_sum;
    
    const int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + row_global) * HEAD_DIM + dim_idx;
    gmem_o[out_offset] = __float2half(final_output);
}

// ============================================================================
// PyTorch Wrapper with Split-K
// ============================================================================
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = torch::empty_like(Q);
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    TORCH_CHECK(head_dim == 128, "This kernel only supports head_dim=128");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    
    const int q_blocks_per_head = div_up(seq_len, BLOCK_ROWS_Q);
    
    // Allocate temporary buffers for partial results
    const int num_partial_blocks = batch_size * num_heads * q_blocks_per_head * SPLIT_K;
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    
    auto partial_o = torch::empty({num_partial_blocks, BLOCK_ROWS_Q, HEAD_DIM}, options_float);
    auto partial_max = torch::empty({num_partial_blocks, BLOCK_ROWS_Q}, options_float);
    auto partial_sum = torch::empty({num_partial_blocks, BLOCK_ROWS_Q}, options_float);
    
    // Shared memory size
    const int smem_size = max(BLOCK_ROWS_Q, 2 * BLOCK_COLS_KV * 2) * HEAD_DIM * sizeof(__half);
    if (smem_size > 48000) {
        cudaFuncSetAttribute(flash_attention_splitk_kernel, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    // Launch with split in grid dimension
    dim3 grid_splitk(q_blocks_per_head, batch_size * num_heads, SPLIT_K);
    dim3 block(THREADS_PER_BLOCK);
    
    // SINGLE kernel launch - all splits run in parallel
    flash_attention_splitk_kernel<<<grid_splitk, block, smem_size>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), 
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()), 
        partial_o.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        batch_size, num_heads, seq_len, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Split-K kernel launch failed: ", cudaGetErrorString(err));

    // Launch reduction kernel
    const int total_output_elements = batch_size * num_heads * q_blocks_per_head * BLOCK_ROWS_Q * HEAD_DIM;
    const int reduce_threads = 256;
    const int reduce_blocks = div_up(total_output_elements, reduce_threads);
    
    flash_attention_reduce_kernel<<<reduce_blocks, reduce_threads>>>(
        partial_o.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        batch_size, num_heads, seq_len);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Reduction kernel launch failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(err));

    return O;
}

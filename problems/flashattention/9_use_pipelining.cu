// flash_attention_wmma_optimized_v7.cu
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/pipeline> // Include for asynchronous pipelining
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>

using namespace nvcuda;

// --- Kernel Parameters (same as v6) ---
#define BLOCK_M 32
#define BLOCK_N 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define THREADS_PER_BLOCK 256

// --- Padding (same as v6) ---
#define SHMEM_HEAD_DIM_PADDED (128 + 8)
constexpr int SHMEM_BLOCK_N_PADDED = BLOCK_N + 8;


// Helper function to perform the async global->shared copy for K and V tiles.
// The compiler will convert this to efficient cp.async instructions.
__device__ void load_kv_tile(
    const half* K_gmem, const half* V_gmem,
    half* k_smem_t, half* v_smem,
    int head_dim, int seq_len, int j_block_start) {
    
    for (int i = threadIdx.x; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (j_block_start + row < seq_len) {
            int gmem_idx = (j_block_start + row) * head_dim + col;
            // Transpose K on the fly while loading
            k_smem_t[col * SHMEM_BLOCK_N_PADDED + row] = K_gmem[gmem_idx];
            // Load V with padding
            v_smem[row * SHMEM_HEAD_DIM_PADDED + col] = V_gmem[gmem_idx];
        } else {
            // Explicitly zero out for correctness, especially for the last block
            k_smem_t[col * SHMEM_BLOCK_N_PADDED + row] = __float2half(0.0f);
            v_smem[row * SHMEM_HEAD_DIM_PADDED + col] = __float2half(0.0f);
        }
    }
}


__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4) wmma_flash_attention_kernel_v7(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale) {

    // --- Shared Memory Allocation (Double Buffered) ---
    extern __shared__ char shmem[];
    size_t offset = 0;

    // --- Non-buffered tiles (persistent throughout the kernel) ---
    half* q_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(half);
    
    float* o_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(float);
    
    float* m_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * sizeof(float);
    
    float* l_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * sizeof(float);

    // --- Double-buffered tiles (for pipelining) ---
    // We create an array of 2 pointers for each tile
    half* k_tiles[2];
    k_tiles[0] = reinterpret_cast<half*>(shmem + offset);
    offset += head_dim * SHMEM_BLOCK_N_PADDED * sizeof(half);
    k_tiles[1] = reinterpret_cast<half*>(shmem + offset);
    offset += head_dim * SHMEM_BLOCK_N_PADDED * sizeof(half);
    
    half* v_tiles[2];
    v_tiles[0] = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half);
    v_tiles[1] = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half);

    float* s_tiles[2];
    s_tiles[0] = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(float);
    s_tiles[1] = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(float);

    half* p_tiles[2];
    p_tiles[0] = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(half);
    p_tiles[1] = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(half);


    // --- Indexing and Initialization ---
    int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;
    
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const half* K_gmem_head_ptr = K + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const half* V_gmem_head_ptr = V + (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Initialize persistent tiles
    for (int i = tid; i < BLOCK_M; i += THREADS_PER_BLOCK) {
        m_tile[i] = -INFINITY;
        l_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * SHMEM_HEAD_DIM_PADDED; i += THREADS_PER_BLOCK) {
        o_tile[i] = 0.0f;
    }

    // Load persistent Q tile
    // (This code is identical to v6)
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            int q_gmem_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
            q_tile[row * SHMEM_HEAD_DIM_PADDED + col] = Q[q_gmem_offset];
        } else {
            q_tile[row * SHMEM_HEAD_DIM_PADDED + col] = __float2half(0.0f);
        }
    }
    __syncthreads();


    // --- Pipelining Main Loop ---
    // Create a pipeline object.
    auto pipeline = cuda::make_pipeline();
    const int main_loop_count = num_kv_blocks;

    // --- Prime the pipeline: Asynchronously load the first K/V block ---
    if (main_loop_count > 0) {
        pipeline.producer_acquire();
        load_kv_tile(K_gmem_head_ptr, V_gmem_head_ptr, k_tiles[0], v_tiles[0], head_dim, seq_len, 0);
        pipeline.producer_commit();
    }

    for (int j = 0; j < main_loop_count; ++j) {
        int current_stage = j % 2;
        int next_stage = (j + 1) % 2;

        // --- Load Next K/V Block (Producer Stage) ---
        // Asynchronously load data for iteration j+1 while we compute on j
        if (j + 1 < main_loop_count) {
            pipeline.producer_acquire();
            int j_block_start_next = (j + 1) * BLOCK_N;
            load_kv_tile(K_gmem_head_ptr, V_gmem_head_ptr, k_tiles[next_stage], v_tiles[next_stage], head_dim, seq_len, j_block_start_next);
            pipeline.producer_commit();
        }

        // --- Wait for Current Data and Compute (Consumer Stage) ---
        pipeline.consumer_wait();
        
        int j_block_start = j * BLOCK_N;

        // The entire compute logic from v6 goes here, but uses the double-buffered pointers
        // (k_tiles[current_stage], v_tiles[current_stage], etc.)

        // === Step 1: Compute S = Q * K^T ===
        {
            // ... this logic is identical to v6 ...
            const int num_s_tiles_n = BLOCK_N / WMMA_N;
            int s_row_tile_idx = warp_id / num_s_tiles_n;
            int s_col_tile_idx = warp_id % num_s_tiles_n;
            int s_row_start = s_row_tile_idx * WMMA_M;
            int s_col_start = s_col_tile_idx * WMMA_N;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_k;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
            wmma::fill_fragment(frag_s, 0.0f);

            for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
                half* q_ptr = &q_tile[s_row_start * SHMEM_HEAD_DIM_PADDED + k_step];
                half* k_ptr = &k_tiles[current_stage][k_step * SHMEM_BLOCK_N_PADDED + s_col_start]; // Use buffered pointer

                wmma::load_matrix_sync(frag_q, q_ptr, SHMEM_HEAD_DIM_PADDED);
                wmma::load_matrix_sync(frag_k, k_ptr, SHMEM_BLOCK_N_PADDED);
                wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
            }
            float* s_ptr = &s_tiles[current_stage][s_row_start * SHMEM_BLOCK_N_PADDED + s_col_start]; // Use buffered pointer
            wmma::store_matrix_sync(s_ptr, frag_s, SHMEM_BLOCK_N_PADDED, wmma::mem_row_major);
        }
        __syncthreads();

        // === Step 2: Softmax ===
        {
            // ... this logic is identical to v6 ...
            const int lane_id = tid % 32;
            const int num_warps = THREADS_PER_BLOCK / 32;
            for (int row = warp_id; row < BLOCK_M; row += num_warps) {
                if (block_row_start + row >= seq_len) continue;
                float m_prev = m_tile[row];
                float l_prev = l_tile[row];
                
                float m_new = -INFINITY;
                for (int j_s = lane_id; j_s < BLOCK_N; j_s += 32) {
                    if (j_block_start + j_s < seq_len) {
                        m_new = fmaxf(m_new, s_tiles[current_stage][row * SHMEM_BLOCK_N_PADDED + j_s] * scale);
                    }
                }
                for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset_shfl));
                m_new = __shfl_sync(0xffffffff, m_new, 0);

                float m_curr = fmaxf(m_prev, m_new);
                float exp_m_diff = expf(m_prev - m_curr);
                for (int d = lane_id; d < head_dim; d += 32) {
                    o_tile[row * SHMEM_HEAD_DIM_PADDED + d] *= exp_m_diff;
                }
                
                float l_new = 0.0f;
                for (int j_p = lane_id; j_p < BLOCK_N; j_p += 32) {
                    float p_val = (j_block_start + j_p < seq_len) ? expf(s_tiles[current_stage][row * SHMEM_BLOCK_N_PADDED + j_p] * scale - m_curr) : 0.0f;
                    l_new += p_val;
                    p_tiles[current_stage][row * SHMEM_BLOCK_N_PADDED + j_p] = __float2half(p_val);
                }
                for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) l_new += __shfl_down_sync(0xffffffff, l_new, offset_shfl);
                l_new = __shfl_sync(0xffffffff, l_new, 0);
                
                float l_curr = l_prev * exp_m_diff + l_new;
                
                if (lane_id == 0) {
                    m_tile[row] = m_curr;
                    l_tile[row] = l_curr;
                }
            }
        }
        __syncthreads();

        // === Step 3: Compute O += P * V ===
        {
            // ... this logic is identical to v6 ...
            const int warps_per_block = THREADS_PER_BLOCK / warpSize;
            const int num_o_tiles_n = head_dim / WMMA_N;
            const int num_o_tiles = (BLOCK_M / WMMA_M) * num_o_tiles_n;
            
            for (int tile_idx = warp_id; tile_idx < num_o_tiles; tile_idx += warps_per_block) {
                int o_row_tile_idx = tile_idx / num_o_tiles_n;
                int o_col_tile_idx = tile_idx % num_o_tiles_n;
                int o_row_start = o_row_tile_idx * WMMA_M;
                int o_col_start = o_col_tile_idx * WMMA_N;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_p;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_v;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_o;

                float* o_ptr = &o_tile[o_row_start * SHMEM_HEAD_DIM_PADDED + o_col_start];
                wmma::load_matrix_sync(frag_o, o_ptr, SHMEM_HEAD_DIM_PADDED, wmma::mem_row_major);

                for (int k_step = 0; k_step < BLOCK_N; k_step += WMMA_K) {
                    half* p_ptr = &p_tiles[current_stage][o_row_start * SHMEM_BLOCK_N_PADDED + k_step];
                    half* v_ptr = &v_tiles[current_stage][k_step * SHMEM_HEAD_DIM_PADDED + o_col_start];
                    
                    wmma::load_matrix_sync(frag_p, p_ptr, SHMEM_BLOCK_N_PADDED);
                    wmma::load_matrix_sync(frag_v, v_ptr, SHMEM_HEAD_DIM_PADDED);
                    wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
                }
                wmma::store_matrix_sync(o_ptr, frag_o, SHMEM_HEAD_DIM_PADDED, wmma::mem_row_major);
            }
        }
        
        // Signal that the consumer stage is done for this iteration.
        pipeline.consumer_release();
    }


    // --- Final Normalization and Write ---
    __syncthreads(); // Make sure all O tile updates are finished
    const int o_elems_to_write = BLOCK_M * head_dim;
    for (int i = tid; i < o_elems_to_write; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            if (l_tile[row] > 0.0f) { // Avoid division by zero
                float inv_l = 1.0f / l_tile[row];
                int o_gmem_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
                int o_shmem_idx = row * SHMEM_HEAD_DIM_PADDED + col;
                O[o_gmem_offset] = __float2half(o_tile[o_shmem_idx] * inv_l);
            }
        }
    }
}

// --- C++ Wrapper Function for PyTorch ---
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = torch::empty_like(Q);

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    TORCH_CHECK(head_dim == 128, "This kernel is hardcoded for head_dim=128");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    
    int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 blocks(num_q_blocks, batch_size * num_heads);
    dim3 threads(THREADS_PER_BLOCK);

    // --- UPDATED SHARED MEMORY CALCULATION ---
    size_t shmem_persistent =
        (BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(half)) +  // q_tile
        (BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(float)) + // o_tile
        (BLOCK_M * 2 * sizeof(float));                      // m/l_tiles

    size_t shmem_per_buffer =
        (head_dim * SHMEM_BLOCK_N_PADDED * sizeof(half)) +  // k_tile_t
        (BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half)) +  // v_tile
        (BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(float)) +  // s_tile
        (BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(half));   // p_tile
        
    size_t shmem_size = shmem_persistent + 2 * shmem_per_buffer;

    cudaFuncSetAttribute(wmma_flash_attention_kernel_v7,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaFuncSetAttribute(wmma_flash_attention_kernel_v7,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    wmma_flash_attention_kernel_v7<<<blocks, threads, shmem_size>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        batch_size, num_heads, seq_len, head_dim, scale
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(err));

    return O;
}


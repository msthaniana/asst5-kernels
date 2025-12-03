// flash_attention_wmma_optimized_v6.cu
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>

using namespace nvcuda;

// --- Kernel Parameters ---
#define BLOCK_M 64
#define BLOCK_N 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define THREADS_PER_BLOCK 512

// --- THE FIX: PADDING FOR SHARED MEMORY ---
// Pad head_dim to avoid 32-way bank conflicts. Stride of 128 is disastrous.
// 128 + 8 = 136. `136*2=272` bytes. `272/4 = 68`. `68 % 32 = 4`. This is a 4-way conflict,
// which is vastly better than a 32-way conflict.
#define SHMEM_HEAD_DIM_PADDED (128 + 8)

__global__ void __launch_bounds__(THREADS_PER_BLOCK) wmma_flash_attention_kernel_v6(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale) {

    extern __shared__ char shmem[];
    size_t offset = 0;
    
    // Use the PADDED stride for memory layout
    half* q_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(half);
    
    // K is transposed, so its shared memory stride is BLOCK_N, which is 64.
    // 64 * sizeof(half) = 128 bytes. 128/4=32. This is also a bank conflict!
    // We should pad this as well. Let's use BLOCK_N + 8 = 72.
    constexpr int SHMEM_BLOCK_N_PADDED = BLOCK_N + 8;

    half* k_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half);  // [BLOCK_N][head_dim]
    
    half* v_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half);

    float* s_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(float);
    
    half* p_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(half);
    
    float* o_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(float);
    
    float* m_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * sizeof(float);
    float* l_tile = reinterpret_cast<float*>(shmem + offset);

    int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;

    // --- Initialization ---
    for (int i = tid; i < BLOCK_M; i += THREADS_PER_BLOCK) {
        m_tile[i] = -INFINITY;
        l_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * SHMEM_HEAD_DIM_PADDED; i += THREADS_PER_BLOCK) {
        o_tile[i] = 0.0f;
    }

    // --- Load Q Tile ---
    const int q_elems_to_load = BLOCK_M * head_dim;
    for (int i = tid; i < q_elems_to_load; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            int q_gmem_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
            int q_shmem_idx = row * SHMEM_HEAD_DIM_PADDED + col;
            q_tile[q_shmem_idx] = Q[q_gmem_offset];
        } else {
            // Zero out to avoid uninitialized data in padding
            int q_shmem_idx = row * SHMEM_HEAD_DIM_PADDED + col;
            q_tile[q_shmem_idx] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // --- Main Loop ---
    for (int j_block_start = 0; j_block_start < seq_len; j_block_start += BLOCK_N) {
        // --- Load K and V Tiles ---
        const int kv_elems_to_load = BLOCK_N * head_dim;
        for (int i = tid; i < kv_elems_to_load; i += THREADS_PER_BLOCK) {
            int row = i / head_dim;  // 0 to BLOCK_N-1
            int col = i % head_dim;  // 0 to 127
            
            if (j_block_start + row < seq_len) {
                int kv_gmem_offset = ((batch_idx * num_heads + head_idx) * seq_len + (j_block_start + row)) * head_dim + col;
                
                // Store K and V with same layout: [row][col]
                k_tile[row * SHMEM_HEAD_DIM_PADDED + col] = K[kv_gmem_offset];
                v_tile[row * SHMEM_HEAD_DIM_PADDED + col] = V[kv_gmem_offset];
            } else {
                k_tile[row * SHMEM_HEAD_DIM_PADDED + col] = __float2half(0.0f);
                v_tile[row * SHMEM_HEAD_DIM_PADDED + col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // === Step 1: Compute S = Q * K^T ===
        {
            const int num_s_tiles_m = BLOCK_M / WMMA_M;  // 8 tiles
            const int num_s_tiles_n = BLOCK_N / WMMA_N;  // 4 tiles
            const int total_tiles = num_s_tiles_m * num_s_tiles_n;  // 32 tiles
            
            for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += (THREADS_PER_BLOCK / 32)) {
                int s_row_tile_idx = tile_idx / num_s_tiles_n;
                int s_col_tile_idx = tile_idx % num_s_tiles_n;
                int s_row_start = s_row_tile_idx * WMMA_M;
                int s_col_start = s_col_tile_idx * WMMA_N;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;  // col_major!
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
                wmma::fill_fragment(frag_s, 0.0f);

                for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
                    // Q: Load row s_row_start, starting at column k_step
                    half* q_ptr = &q_tile[s_row_start * SHMEM_HEAD_DIM_PADDED + k_step];
                    
                    // K^T: We want columns s_col_start to s_col_start+15
                    // Since K is stored as [BLOCK_N][head_dim], row s_col_start gives us what we need
                    half* k_ptr = &k_tile[s_col_start * SHMEM_HEAD_DIM_PADDED + k_step];

                    wmma::load_matrix_sync(frag_q, q_ptr, SHMEM_HEAD_DIM_PADDED);
                    wmma::load_matrix_sync(frag_k, k_ptr, SHMEM_HEAD_DIM_PADDED);  // Loaded as col_major
                    wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
                }
                
                float* s_ptr = &s_tile[s_row_start * SHMEM_BLOCK_N_PADDED + s_col_start];
                wmma::store_matrix_sync(s_ptr, frag_s, SHMEM_BLOCK_N_PADDED, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // === Step 2: Softmax ===
        {
            const int lane_id = tid % 32;
            const int num_warps = THREADS_PER_BLOCK / 32;
            for (int row = warp_id; row < BLOCK_M; row += num_warps) {
                if (block_row_start + row >= seq_len) continue;
                float m_prev = m_tile[row];
                float l_prev = l_tile[row];
                
                float m_new = -INFINITY;
                for (int j = lane_id; j < BLOCK_N; j += 32) {
                    if (j_block_start + j < seq_len) {
                        m_new = fmaxf(m_new, s_tile[row * SHMEM_BLOCK_N_PADDED + j] * scale);
                    }
                }
                for (int offset = 16; offset > 0; offset /= 2) m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
                m_new = __shfl_sync(0xffffffff, m_new, 0);

                float m_curr = fmaxf(m_prev, m_new);
                float exp_m_diff = expf(m_prev - m_curr);
                for (int d = lane_id; d < head_dim; d += 32) {
                    o_tile[row * SHMEM_HEAD_DIM_PADDED + d] *= exp_m_diff;
                }
                
                float l_new = 0.0f;
                for (int j = lane_id; j < BLOCK_N; j += 32) {
                    float p_val = (j_block_start + j < seq_len) ? expf(s_tile[row * SHMEM_BLOCK_N_PADDED + j] * scale - m_curr) : 0.0f;
                    l_new += p_val;
                    p_tile[row * SHMEM_BLOCK_N_PADDED + j] = __float2half(p_val);
                }
                for (int offset = 16; offset > 0; offset /= 2) l_new += __shfl_down_sync(0xffffffff, l_new, offset);
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
                    half* p_ptr = &p_tile[o_row_start * SHMEM_BLOCK_N_PADDED + k_step];
                    half* v_ptr = &v_tile[k_step * SHMEM_HEAD_DIM_PADDED + o_col_start];
                    
                    wmma::load_matrix_sync(frag_p, p_ptr, SHMEM_BLOCK_N_PADDED);
                    wmma::load_matrix_sync(frag_v, v_ptr, SHMEM_HEAD_DIM_PADDED);
                    wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
                }
                wmma::store_matrix_sync(o_ptr, frag_o, SHMEM_HEAD_DIM_PADDED, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // --- Final Write ---
    const int o_elems_to_write = BLOCK_M * head_dim;
    for (int i = tid; i < o_elems_to_write; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            float inv_l = 1.0f / l_tile[row];
            int o_gmem_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
            int o_shmem_idx = row * SHMEM_HEAD_DIM_PADDED + col;
            O[o_gmem_offset] = __float2half(o_tile[o_shmem_idx] * inv_l);
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

    constexpr int SHMEM_BLOCK_N_PADDED = BLOCK_N + 8;
    size_t shmem_size = (BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(half))  // q_tile
                    + (BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half))   // k_tile (CHANGED!)
                    + (BLOCK_N * SHMEM_HEAD_DIM_PADDED * sizeof(half))   // v_tile
                    + (BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(float))   // s_tile
                    + (BLOCK_M * SHMEM_BLOCK_N_PADDED * sizeof(half))    // p_tile
                    + (BLOCK_M * SHMEM_HEAD_DIM_PADDED * sizeof(float))  // o_tile
                    + (BLOCK_M * 2 * sizeof(float));                     // m/l_tiles

    cudaFuncSetAttribute(wmma_flash_attention_kernel_v6,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaFuncSetAttribute(wmma_flash_attention_kernel_v6,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    wmma_flash_attention_kernel_v6<<<blocks, threads, shmem_size>>>(
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


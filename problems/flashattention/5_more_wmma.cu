// flash_attention_wmma_optimized.cu
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <iostream>

using namespace nvcuda;

#define BLOCK_M 32
#define BLOCK_N 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define THREADS_PER_BLOCK 128

__global__ void wmma_flash_attention_kernel_v3(
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
    
    half* q_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * head_dim * sizeof(half);
    
    half* k_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * head_dim * sizeof(half);
    
    half* v_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * head_dim * sizeof(half);
    
    float* s_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * BLOCK_N * sizeof(float);
    
    half* p_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * BLOCK_N * sizeof(half);
    
    float* o_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * head_dim * sizeof(float);
    
    float* m_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * sizeof(float);
    
    float* l_tile = reinterpret_cast<float*>(shmem + offset);
    // offset += BLOCK_M * sizeof(float);  // Last one, no need to update

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;

    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;

    // Initialize
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        o_tile[i] = 0.0f;
    }
    if (tid < BLOCK_M) {
        m_tile[tid] = -INFINITY;
        l_tile[tid] = 0.0f;
    }
    __syncthreads();

    // Load Q tile
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + 
                           (block_row_start + row)) * head_dim + col;
            q_tile[row * head_dim + col] = Q[q_offset];
        } else {
            q_tile[row * head_dim + col] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Main loop over K/V blocks
    for (int j_block_start = 0; j_block_start < seq_len; j_block_start += BLOCK_N) {

        // Load K and V tiles
        for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (j_block_start + row < seq_len) {
                int kv_offset = ((batch_idx * num_heads + head_idx) * seq_len + 
                                (j_block_start + row)) * head_dim + col;
                k_tile[row * head_dim + col] = K[kv_offset];
                v_tile[row * head_dim + col] = V[kv_offset];
            } else {
                k_tile[row * head_dim + col] = __float2half(0.0f);
                v_tile[row * head_dim + col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // === Step 1: Compute S = Q * K^T using WMMA ===
        {
            int warp_m_idx = warp_id / 2;
            int warp_n_idx = warp_id % 2;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s[2];

            wmma::fill_fragment(frag_s[0], 0.0f);
            wmma::fill_fragment(frag_s[1], 0.0f);

            for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
                int q_row = warp_m_idx * WMMA_M;
                int k_col_0 = warp_n_idx * 2 * WMMA_N;
                int k_col_1 = k_col_0 + WMMA_N;

                half* q_ptr = &q_tile[q_row * head_dim + k_step];
                half* k_ptr_0 = &k_tile[k_col_0 * head_dim + k_step];
                half* k_ptr_1 = &k_tile[k_col_1 * head_dim + k_step];

                wmma::load_matrix_sync(frag_q, q_ptr, head_dim);
                wmma::load_matrix_sync(frag_k, k_ptr_0, head_dim);
                wmma::mma_sync(frag_s[0], frag_q, frag_k, frag_s[0]);

                wmma::load_matrix_sync(frag_k, k_ptr_1, head_dim);
                wmma::mma_sync(frag_s[1], frag_q, frag_k, frag_s[1]);
            }

            int s_row = warp_m_idx * WMMA_M;
            int s_col_0 = warp_n_idx * 2 * WMMA_N;
            int s_col_1 = s_col_0 + WMMA_N;

            float* s_ptr_0 = &s_tile[s_row * BLOCK_N + s_col_0];
            float* s_ptr_1 = &s_tile[s_row * BLOCK_N + s_col_1];

            wmma::store_matrix_sync(s_ptr_0, frag_s[0], BLOCK_N, wmma::mem_row_major);
            wmma::store_matrix_sync(s_ptr_1, frag_s[1], BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads();

        // === Step 2: Compute Softmax and create P tile ===
        for (int row = tid; row < BLOCK_M; row += THREADS_PER_BLOCK) {
            if (block_row_start + row >= seq_len) continue;

            float m_prev = m_tile[row];
            float l_prev = l_tile[row];
            
            // Find max in this row
            float m_new = -INFINITY;
            for (int j = 0; j < BLOCK_N; ++j) {
                if (j_block_start + j < seq_len) {
                    float s_val = s_tile[row * BLOCK_N + j] * scale;
                    m_new = fmaxf(m_new, s_val);
                }
            }
            
            float m_curr = fmaxf(m_prev, m_new);
            float exp_m_diff = expf(m_prev - m_curr);
            
            // Rescale existing output accumulator
            for (int d = 0; d < head_dim; ++d) {
                o_tile[row * head_dim + d] *= exp_m_diff;
            }
            
            // Compute P and new sum
            float l_curr = l_prev * exp_m_diff;
            for (int j = 0; j < BLOCK_N; ++j) {
                float p_val;
                if (j_block_start + j < seq_len) {
                    float s_val = s_tile[row * BLOCK_N + j] * scale;
                    p_val = expf(s_val - m_curr);
                    l_curr += p_val;
                } else {
                    p_val = 0.0f;
                }
                p_tile[row * BLOCK_N + j] = __float2half(p_val);
            }
            
            m_tile[row] = m_curr;
            l_tile[row] = l_curr;
        }
        __syncthreads();

        // === Step 3: Compute O += P * V using WMMA ===
        {
            int num_row_tiles = BLOCK_M / WMMA_M;  // 2
            int num_col_tiles = head_dim / WMMA_N;  // 8
            
            int warp_row_tile = warp_id / 2;
            
            for (int col_tile = warp_id % 2; col_tile < num_col_tiles; col_tile += 2) {
                int warp_row_start = warp_row_tile * WMMA_M;
                int warp_col_start = col_tile * WMMA_N;
                
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_p;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_v;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_o;

                float* o_ptr = &o_tile[warp_row_start * head_dim + warp_col_start];
                wmma::load_matrix_sync(frag_o, o_ptr, head_dim, wmma::mem_row_major);

                for (int k_step = 0; k_step < BLOCK_N; k_step += WMMA_K) {
                    half* p_ptr = &p_tile[warp_row_start * BLOCK_N + k_step];
                    half* v_ptr = &v_tile[k_step * head_dim + warp_col_start];
                    
                    wmma::load_matrix_sync(frag_p, p_ptr, BLOCK_N);
                    wmma::load_matrix_sync(frag_v, v_ptr, head_dim);
                    wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
                }

                wmma::store_matrix_sync(o_ptr, frag_o, head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Final normalization and write
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            float inv_l = 1.0f / l_tile[row];
            int o_offset = ((batch_idx * num_heads + head_idx) * seq_len + 
                           (block_row_start + row)) * head_dim + col;
            O[o_offset] = __float2half(o_tile[i] * inv_l);
        }
    }
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto O = torch::empty_like(Q);

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    TORCH_CHECK(head_dim % 16 == 0, "Head dimension must be a multiple of 16");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Must be FP16");

    int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 blocks(num_q_blocks, batch_size * num_heads);
    dim3 threads(THREADS_PER_BLOCK);

    size_t shmem_size = (BLOCK_M * head_dim) * sizeof(half)
                      + (BLOCK_N * head_dim) * sizeof(half)
                      + (BLOCK_N * head_dim) * sizeof(half)
                      + (BLOCK_M * BLOCK_N) * sizeof(float)
                      + (BLOCK_M * BLOCK_N) * sizeof(half)
                      + (BLOCK_M * head_dim) * sizeof(float)
                      + (BLOCK_M * 2) * sizeof(float);

    cudaFuncSetAttribute(wmma_flash_attention_kernel_v3,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaFuncSetAttribute(wmma_flash_attention_kernel_v3,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    wmma_flash_attention_kernel_v3<<<blocks, threads, shmem_size>>>(
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
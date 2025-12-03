// flash_attention_wmma_optimized_v5.cu
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>

using namespace nvcuda;

// --- Kernel Parameters ---
// Block shape for the outer Q loop
#define BLOCK_M 32
// Block shape for the inner K/V loop
#define BLOCK_N 64
// WMMA intrinsic shape
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// Number of threads per block. 256 threads = 8 warps.
#define THREADS_PER_BLOCK 256

__global__ void __launch_bounds__(THREADS_PER_BLOCK) wmma_flash_attention_kernel_v5(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale) {

    // --- Shared Memory Allocation ---
    extern __shared__ char shmem[];
    size_t offset = 0;

    // Tile for Q input block (BLOCK_M x head_dim)
    half* q_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * head_dim * sizeof(half);

    // Tile for K input block, stored transposed (head_dim x BLOCK_N)
    half* k_tile_t = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * head_dim * sizeof(half);

    // Tile for V input block (BLOCK_N x head_dim)
    half* v_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_N * head_dim * sizeof(half);

    // Tile for S = Q*K^T output (BLOCK_M x BLOCK_N), stored as float32
    float* s_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * BLOCK_N * sizeof(float);

    // Tile for P = softmax(S) output (BLOCK_M x BLOCK_N), stored as half
    half* p_tile = reinterpret_cast<half*>(shmem + offset);
    offset += BLOCK_M * BLOCK_N * sizeof(half);

    // Tile for the accumulator O (BLOCK_M x head_dim), stored as float32
    float* o_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * head_dim * sizeof(float);

    // Tiles for row-wise softmax statistics (m_i and l_i)
    float* m_tile = reinterpret_cast<float*>(shmem + offset);
    offset += BLOCK_M * sizeof(float);
    float* l_tile = reinterpret_cast<float*>(shmem + offset);

    // --- Thread and Block Indexing ---
    int tid = threadIdx.x;
    const int warp_id = tid / warpSize;

    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;

    // --- Initialization ---
    // Initialize m, l, and o tiles in parallel. Each thread handles a subset.
    for (int i = tid; i < BLOCK_M; i += THREADS_PER_BLOCK) {
        m_tile[i] = -INFINITY;
        l_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        o_tile[i] = 0.0f;
    }

    // --- Load Q Tile (Coalesced) ---
    // Threads contiguously load the Q tile from global memory to shared memory.
    const int q_elems_per_block = BLOCK_M * head_dim;
    for (int i = tid; i < q_elems_per_block; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
            q_tile[i] = Q[q_offset];
        } else {
            q_tile[i] = __float2half(0.0f);
        }
    }
    __syncthreads(); // Ensure Q tile is fully loaded before use.

    // --- Main Loop over K/V Blocks ---
    for (int j_block_start = 0; j_block_start < seq_len; j_block_start += BLOCK_N) {
        // --- Load K and V Tiles (Coalesced and Transposed) ---
        const int kv_elems_per_block = BLOCK_N * head_dim;
        for (int i = tid; i < kv_elems_per_block; i += THREADS_PER_BLOCK) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (j_block_start + row < seq_len) {
                int kv_offset = ((batch_idx * num_heads + head_idx) * seq_len + (j_block_start + row)) * head_dim + col;
                k_tile_t[col * BLOCK_N + row] = K[kv_offset]; // Transpose on store for efficient WMMA load
                v_tile[i] = V[kv_offset];
            } else {
                k_tile_t[col * BLOCK_N + row] = __float2half(0.0f);
                v_tile[i] = __float2half(0.0f);
            }
        }
        __syncthreads(); // Ensure K/V tiles are loaded.

        // === Step 1: Compute S = Q * K^T using WMMA (v5 FIX) ===
        {
            // The S matrix is (32x64), composed of (16x16) WMMA tiles.
            // This gives a 2x4 grid of 8 tiles. We have 8 warps.
            // We can directly map each warp to compute exactly one tile.
            const int num_s_tiles_n = BLOCK_N / WMMA_N; // 64 / 16 = 4

            int s_row_tile_idx = warp_id / num_s_tiles_n; // Warp 0-3 -> 0; Warp 4-7 -> 1
            int s_col_tile_idx = warp_id % num_s_tiles_n; // Warp 0,4 -> 0; 1,5 -> 1; etc.

            int s_row_start = s_row_tile_idx * WMMA_M;
            int s_col_start = s_col_tile_idx * WMMA_N;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_k; // row_major because k_tile_t is pre-transposed
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
            wmma::fill_fragment(frag_s, 0.0f);

            for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
                half* q_ptr = &q_tile[s_row_start * head_dim + k_step];
                half* k_ptr = &k_tile_t[k_step * BLOCK_N + s_col_start];

                wmma::load_matrix_sync(frag_q, q_ptr, head_dim);
                wmma::load_matrix_sync(frag_k, k_ptr, BLOCK_N); // Stride is BLOCK_N for transposed layout
                wmma::mma_sync(frag_s, frag_q, frag_k, frag_s);
            }

            float* s_ptr = &s_tile[s_row_start * BLOCK_N + s_col_start];
            wmma::store_matrix_sync(s_ptr, frag_s, BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads(); // Ensure S is fully computed.

        // === Step 2: Compute Softmax and create P tile ===
        {
            // Each warp processes a subset of rows from the S/P tiles.
            // This is a correct parallel pattern as rows are independent.
            const int lane_id = tid % 32;
            const int num_warps = THREADS_PER_BLOCK / 32;

            for (int row = warp_id; row < BLOCK_M; row += num_warps) {
                if (block_row_start + row >= seq_len) continue;

                float m_prev = m_tile[row];
                float l_prev = l_tile[row];

                // Find max of current S tile row (m_new)
                float m_new = -INFINITY;
                for (int j = lane_id; j < BLOCK_N; j += 32) {
                    if (j_block_start + j < seq_len) {
                        m_new = fmaxf(m_new, s_tile[row * BLOCK_N + j] * scale);
                    }
                }
                for (int offset = 16; offset > 0; offset /= 2) m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
                m_new = __shfl_sync(0xffffffff, m_new, 0);

                // Update overall max (m_curr) and rescale O and l
                float m_curr = fmaxf(m_prev, m_new);
                float exp_m_diff = expf(m_prev - m_curr);
                for (int d = lane_id; d < head_dim; d += 32) o_tile[row * head_dim + d] *= exp_m_diff;
                
                // Calculate P and sum of new exponentials (l_new)
                float l_new = 0.0f;
                for (int j = lane_id; j < BLOCK_N; j += 32) {
                    float p_val = (j_block_start + j < seq_len) ? expf(s_tile[row * BLOCK_N + j] * scale - m_curr) : 0.0f;
                    l_new += p_val;
                    p_tile[row * BLOCK_N + j] = __float2half(p_val);
                }
                for (int offset = 16; offset > 0; offset /= 2) l_new += __shfl_down_sync(0xffffffff, l_new, offset);
                l_new = __shfl_sync(0xffffffff, l_new, 0);
                
                // Update overall sum (l_curr)
                float l_curr = l_prev * exp_m_diff + l_new;

                if (lane_id == 0) { // Only one thread per row needs to update stats
                    m_tile[row] = m_curr;
                    l_tile[row] = l_curr;
                }
            }
        }
        __syncthreads(); // Ensure P, m, and l are updated.

        // === Step 3: Compute O += P * V using WMMA ===
        {
            // O is a (32x128) tile, which is a 2x8 grid of (16x16) WMMA tiles = 16 tiles total.
            // We have 8 warps, so each warp computes 2 tiles. The grid-stride loop is a
            // robust way to distribute this work.
            const int warps_per_block = THREADS_PER_BLOCK / warpSize;
            const int num_o_tiles_m = BLOCK_M / WMMA_M;
            const int num_o_tiles_n = head_dim / WMMA_N;
            const int num_o_tiles = num_o_tiles_m * num_o_tiles_n;

            for (int tile_idx = warp_id; tile_idx < num_o_tiles; tile_idx += warps_per_block) {
                int o_row_tile_idx = tile_idx / num_o_tiles_n;
                int o_col_tile_idx = tile_idx % num_o_tiles_n;

                int o_row_start = o_row_tile_idx * WMMA_M;
                int o_col_start = o_col_tile_idx * WMMA_N;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_p;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_v;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_o;

                float* o_ptr = &o_tile[o_row_start * head_dim + o_col_start];
                wmma::load_matrix_sync(frag_o, o_ptr, head_dim, wmma::mem_row_major);

                for (int k_step = 0; k_step < BLOCK_N; k_step += WMMA_K) {
                    half* p_ptr = &p_tile[o_row_start * BLOCK_N + k_step];
                    half* v_ptr = &v_tile[k_step * head_dim + o_col_start];

                    wmma::load_matrix_sync(frag_p, p_ptr, BLOCK_N);
                    wmma::load_matrix_sync(frag_v, v_ptr, head_dim);
                    wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
                }
                wmma::store_matrix_sync(o_ptr, frag_o, head_dim, wmma::mem_row_major);
            }
        }
        __syncthreads(); // Ensure O is updated before the next outer loop iteration.
    }

    // --- Final Normalization and Write to Global Memory (Coalesced) ---
    const int o_elems_per_block = BLOCK_M * head_dim;
    for (int i = tid; i < o_elems_per_block; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            float inv_l = 1.0f / l_tile[row];
            int o_offset = ((batch_idx * num_heads + head_idx) * seq_len + (block_row_start + row)) * head_dim + col;
            O[o_offset] = __float2half(o_tile[i] * inv_l);
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

    TORCH_CHECK(head_dim == 128, "This kernel is optimized for head_dim=128");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "Inputs must be FP16");
    
    int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 blocks(num_q_blocks, batch_size * num_heads);
    dim3 threads(THREADS_PER_BLOCK);

    // Calculate required shared memory
    size_t shmem_size = (BLOCK_M * head_dim * sizeof(half))   // q_tile
                      + (BLOCK_N * head_dim * sizeof(half))   // k_tile_t
                      + (BLOCK_N * head_dim * sizeof(half))   // v_tile
                      + (BLOCK_M * BLOCK_N * sizeof(float))   // s_tile
                      + (BLOCK_M * BLOCK_N * sizeof(half))    // p_tile
                      + (BLOCK_M * head_dim * sizeof(float))  // o_tile
                      + (BLOCK_M * 2 * sizeof(float));        // m_tile and l_tile

    // Request dynamic shared memory from CUDA runtime
    cudaFuncSetAttribute(wmma_flash_attention_kernel_v5,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaFuncSetAttribute(wmma_flash_attention_kernel_v5,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Launch the kernel
    wmma_flash_attention_kernel_v5<<<blocks, threads, shmem_size>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        batch_size, num_heads, seq_len, head_dim, scale
    );

    // Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    
    // It's good practice to check after sync as well for async errors
    // cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "Kernel execution failed: ", cudaGetErrorString(err));

    return O;
}



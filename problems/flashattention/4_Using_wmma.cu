// flash_attention_wmma.cu
// Kernel leveraging WMMA (Tensor Cores) for high performance.

#include <cuda_fp16.h> // Required for __half
#include <mma.h>       // Required for WMMA
#include <torch/extension.h>

using namespace nvcuda; // Use the wmma namespace

// --- Tiling and WMMA Configuration ---
// Dimensions of the tiles processed by a THREAD BLOCK
#define BLOCK_M 32
#define BLOCK_N 64
// Dimensions of the matrix tile processed by a single WARP using WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// Number of threads in the block. Must be a multiple of 32.
#define THREADS_PER_BLOCK 128

// ------------------------------------------------------------------------
// WMMA-based Tiled FlashAttention CUDA Kernel
// ------------------------------------------------------------------------
__global__ void wmma_flash_attention_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {

    // --- Shared Memory ---
    // Use dynamic shared memory, partitioned manually.
    extern __shared__ char shmem[];
    half* q_tile = reinterpret_cast<half*>(shmem);
    half* k_tile = q_tile + BLOCK_M * head_dim;
    half* v_tile = k_tile + BLOCK_N * head_dim;
    // NEW: A shared memory tile to store the S = Q*K^T results
    float* s_tile = reinterpret_cast<float*>(v_tile + BLOCK_N * head_dim);

    // --- Thread and Warp Identification ---
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    // --- NEW: Map Warps to a 2D grid to process the S tile ---
    // A thread block computes a 32x64 S-tile. We have 4 warps (128 threads).
    // We can arrange them in a 2x2 grid, where each warp computes two 16x16 sub-tiles.
    // warp_m_idx determines the row-block (0 or 1) of the S-tile this warp works on.
    // warp_n_idx determines the column-block (0, 1, 2, or 3)
    int warp_m_idx = warp_id / 2; // (0 or 1)
    int warp_n_idx = (warp_id % 2) * 2; // (0 or 2)

    // --- Map block to (batch, head, token_block) ---
    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;

    // --- Load Q Tile into Shared Memory (Once per block) ---
    // All 128 threads cooperate to load a 32xhead_dim tile of Q.
    for (int i = tid; i < BLOCK_M * head_dim; i += THREADS_PER_BLOCK) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (block_row_start + row < seq_len) {
            int q_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + (block_row_start + row)) * head_dim + col;
            q_tile[row * head_dim + col] = Q[q_offset];
        } else {
            q_tile[row * head_dim + col] = __float2half(0.0f);
        }
    }

    // --- Per-thread local state for the final softmax and output accumulation ---
    // We go back to a "1 thread per Q row" model for this final part.
    // Each thread handles ONE row of the 32x64 output tile.
    float o_acc[128]; // This can be large, risk of register spilling //head_dim=128 hardcoded here
    float m_i = -INFINITY;
    float l_i = 0.0f;

    if (tid < BLOCK_M) {
        for (int d = 0; d < head_dim; ++d) { o_acc[d] = 0.0f; }
    }
    __syncthreads(); // Ensure Q tile is fully loaded

    // --- Main Loop: Iterate over K and V Tiles ---
    for (int j_block_start = 0; j_block_start < seq_len; j_block_start += BLOCK_N) {

        // --- Load K and V Tiles into Shared Memory ---
        for (int i = tid; i < BLOCK_N * head_dim; i += THREADS_PER_BLOCK) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (j_block_start + row < seq_len) {
                int kv_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + (j_block_start + row)) * head_dim + col;
                k_tile[row * head_dim + col] = K[kv_offset];
                v_tile[row * head_dim + col] = V[kv_offset];
            } else {
                k_tile[row * head_dim + col] = __float2half(0.0f);
                v_tile[row * head_dim + col] = __float2half(0.0f);
            }
        }
        __syncthreads(); // Ensure K, V tiles are loaded

        // --- BOTTLENECK FIX: Compute S_tile = Q_tile * K_tile^T using WMMA ---
        // Each warp computes two 16x16 sub-tiles of the 32x64 S-tile.
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k[2];
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s[2];

        // Initialize accumulator fragments to zero
        wmma::fill_fragment(frag_s[0], 0.0f);
        wmma::fill_fragment(frag_s[1], 0.0f);

        // Loop over the K-dimension of Q and K tiles to compute the dot product
        for (int k_step = 0; k_step < head_dim; k_step += WMMA_K) {
            // Pointers to the start of the sub-tiles this warp will process
            int q_sub_tile_row = warp_m_idx * WMMA_M;
            int k_sub_tile_row[2] = {warp_n_idx * WMMA_N, (warp_n_idx + 1) * WMMA_N};

            half* q_ptr = &q_tile[q_sub_tile_row * head_dim + k_step];
            half* k_ptr[2] = {&k_tile[k_sub_tile_row[0] * head_dim + k_step],
                              &k_tile[k_sub_tile_row[1] * head_dim + k_step]};

            // Load matrices from shared memory into fragments
            wmma::load_matrix_sync(frag_q, q_ptr, head_dim);
            wmma::load_matrix_sync(frag_k[0], k_ptr[0], head_dim);
            wmma::load_matrix_sync(frag_k[1], k_ptr[1], head_dim);

            // Perform matrix multiply-accumulate: S = Q * K^T + S
            wmma::mma_sync(frag_s[0], frag_q, frag_k[0], frag_s[0]);
            wmma::mma_sync(frag_s[1], frag_q, frag_k[1], frag_s[1]);
        }

        // Store the resulting S-tile fragments from registers to shared memory
        int s_sub_tile_row = warp_m_idx * WMMA_M;
        int s_sub_tile_col[2] = {warp_n_idx * WMMA_N, (warp_n_idx + 1) * WMMA_N};
        float* s_ptr[2] = {&s_tile[s_sub_tile_row * BLOCK_N + s_sub_tile_col[0]],
                           &s_tile[s_sub_tile_row * BLOCK_N + s_sub_tile_col[1]]};
        
        wmma::store_matrix_sync(s_ptr[0], frag_s[0], BLOCK_N, wmma::mem_row_major);
        wmma::store_matrix_sync(s_ptr[1], frag_s[1], BLOCK_N, wmma::mem_row_major);
        __syncthreads(); // Wait for all warps to finish writing S-tile

        // --- Online Softmax using the computed S-tile ---
        // Now, threads 0-31 work on their respective rows (0-31) of the S-tile.
        if (tid < BLOCK_M) {
            float m_prev = m_i;
            float m_new = -INFINITY;
            // Find new max in the current S-tile row
            for(int j = 0; j < BLOCK_N; ++j) {
                if (j_block_start + j < seq_len) {
                    m_new = fmaxf(m_new, s_tile[tid * BLOCK_N + j] * scale);
                }
            }
            m_i = fmaxf(m_prev, m_new);
            
            float p_scale = expf(m_prev - m_i);
            l_i = l_i * p_scale;

            // Rescale accumulator
            for (int d = 0; d < head_dim; ++d) { o_acc[d] *= p_scale; }

            // Update accumulator
            for (int j = 0; j < BLOCK_N; ++j) {
                if (j_block_start + j < seq_len) {
                    float p_ij = expf(s_tile[tid * BLOCK_N + j] * scale - m_i);
                    l_i += p_ij;
                    for (int d = 0; d < head_dim; ++d) {
                        o_acc[d] += p_ij * __half2float(v_tile[j * head_dim + d]);
                    }
                }
            }
        }
        __syncthreads(); // Sync before loading next K/V tile
    }

    // --- Final Normalization and Write to Global Memory ---
    if (tid < BLOCK_M && block_row_start + tid < seq_len) {
        float inv_l_i = 1.0f / l_i;
        int o_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + (block_row_start + tid)) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            O[o_offset + d] = __float2half(o_acc[d] * inv_l_i);
        }
    }
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  TORCH_CHECK(head_dim % 16 == 0, "Head dimension must be a multiple of 16 for WMMA.");
  TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Input tensors must be FP16 for this WMMA kernel.");

  int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 blocks(num_q_blocks, batch_size * num_heads);
  dim3 threads(THREADS_PER_BLOCK);

  size_t shmem_size = (BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_N * head_dim) * sizeof(half)
                    + (BLOCK_M * BLOCK_N) * sizeof(float);
  
  // Opt-in to more shared memory than the default 48KB.
  cudaFuncSetAttribute(wmma_flash_attention_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shmem_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "wmma_flash_attention_kernel", ([&] {
        if (std::is_same<scalar_t, c10::Half>::value) {
            wmma_flash_attention_kernel<<<blocks, threads, shmem_size>>>(
                (const half*)Q.data_ptr<scalar_t>(), (const half*)K.data_ptr<scalar_t>(),
                (const half*)V.data_ptr<scalar_t>(), (half*)O.data_ptr<scalar_t>(),
                batch_size, num_heads, seq_len, head_dim, scale);
        } else {
            TORCH_CHECK(false, "This kernel only supports FP16 inputs.");
        }
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  return O;
}
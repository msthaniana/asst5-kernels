// flash_attention_corrected.cu
// Tiled FlashAttention with corrected loading stride.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>

// Tiling Configuration
#define BLOCK_M 32
#define BLOCK_N 32
#define THREADS_PER_BLOCK 128 // Can be > BLOCK_M, but inefficient
#define MAX_HEAD_DIM 128

template <typename scalar_t>
__global__ void tiled_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {

    __shared__ float q_tile[BLOCK_M * MAX_HEAD_DIM];
    __shared__ float k_tile[BLOCK_N * MAX_HEAD_DIM];
    __shared__ float v_tile[BLOCK_N * MAX_HEAD_DIM];

    int tid = threadIdx.x;
    int block_row_start = blockIdx.x * BLOCK_M;
    int head_idx = blockIdx.y % num_heads;
    int batch_idx = blockIdx.y / num_heads;

    // Early exit for threads that are not assigned to a Q row.
    // Only threads 0 to (BLOCK_M - 1) do the work.
    if (tid >= BLOCK_M) return;

    int q_row = block_row_start + tid;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[MAX_HEAD_DIM];

    for (int d = 0; d < head_dim; ++d) {
        o_acc[d] = 0.0f;
    }

    // --- Load Q Tile into Shared Memory ---
    // CORRECTED: Stride is BLOCK_M (the number of participating threads)
    for (int d = tid; d < BLOCK_M * head_dim; d += BLOCK_M) {
        int row = d / head_dim;
        int col = d % head_dim;
        if (block_row_start + row < seq_len) {
            int q_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + (block_row_start + row)) * head_dim + col;
            q_tile[row * head_dim + col] = static_cast<float>(Q[q_offset]);
        }
    }
    __syncthreads();

    for (int j_block_start = 0; j_block_start < seq_len; j_block_start += BLOCK_N) {
        // --- Load K and V Tiles into Shared Memory ---
        // CORRECTED: Stride is BLOCK_M
        for (int d = tid; d < BLOCK_N * head_dim; d += BLOCK_M) {
            int row = d / head_dim;
            int col = d % head_dim;
            if (j_block_start + row < seq_len) {
                int kv_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + (j_block_start + row)) * head_dim + col;
                k_tile[row * head_dim + col] = static_cast<float>(K[kv_offset]);
                v_tile[row * head_dim + col] = static_cast<float>(V[kv_offset]);
            }
        }
        __syncthreads();

        // (The rest of the kernel was logically correct and does not need changes)
        float m_i_new = -INFINITY;
        for (int j = 0; j < BLOCK_N; ++j) {
            int k_row = j_block_start + j;
            if (k_row >= seq_len || q_row >= seq_len) continue;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_tile[tid * head_dim + d] * k_tile[j * head_dim + d];
            }
            score *= scale;
            m_i_new = fmaxf(m_i_new, score);
        }

        float m_prev = m_i;
        m_i = fmaxf(m_prev, m_i_new);
        float p_scale = expf(m_prev - m_i);
        l_i = l_i * p_scale;
        for (int d = 0; d < head_dim; ++d) {
            o_acc[d] *= p_scale;
        }

        for (int j = 0; j < BLOCK_N; ++j) {
            int k_row = j_block_start + j;
            if (k_row >= seq_len || q_row >= seq_len) continue;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_tile[tid * head_dim + d] * k_tile[j * head_dim + d];
            }
            score *= scale;
            float p_ij = expf(score - m_i);
            l_i += p_ij;
            for (int d = 0; d < head_dim; ++d) {
                o_acc[d] += p_ij * v_tile[j * head_dim + d];
            }
        }
        __syncthreads();
    }

    if (q_row < seq_len) {
        float inv_l_i = 1.0f / l_i;
        int o_offset = (batch_idx * num_heads * seq_len + head_idx * seq_len + q_row) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            O[o_offset + d] = static_cast<scalar_t>(o_acc[d] * inv_l_i);
        }
    }
}

// ------------------------------------------------------------------------
// C++ / Python Interface (NO CHANGES NEEDED HERE)
// ------------------------------------------------------------------------
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
  dim3 blocks(num_q_blocks, batch_size * num_heads);
  dim3 threads(THREADS_PER_BLOCK);

  if (head_dim > MAX_HEAD_DIM) {
      throw std::runtime_error("head_dim exceeds MAX_HEAD_DIM defined in the kernel.");
  }
  if (THREADS_PER_BLOCK < BLOCK_M) {
      throw std::runtime_error("This kernel requires THREADS_PER_BLOCK >= BLOCK_M");
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "tiled_flash_attention_kernel", ([&] {
        tiled_flash_attention_kernel<scalar_t><<<blocks, threads>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
            num_heads, seq_len, head_dim, scale);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  return O;
}
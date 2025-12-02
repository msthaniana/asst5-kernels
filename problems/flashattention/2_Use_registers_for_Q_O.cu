// flash_attention_optimized.cu
// Optimized FlashAttention with proper thread utilization for H100

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Constants
#define TILE_SIZE 16
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// ------------------------------------------------------------------------
// Warp-level reduction utilities
// ------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ------------------------------------------------------------------------
// Block-level reduction utilities
// ------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Store warp results
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_max(val);
    
    // Store warp results
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
    }
    
    return val;
}

// ------------------------------------------------------------------------
// Optimized FlashAttention CUDA Kernel
// ------------------------------------------------------------------------
template <typename scalar_t>
__global__ void optimized_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
    
    // Shared memory for reductions
    __shared__ float shared_mem[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_m_i;
    __shared__ float shared_l_i;
    __shared__ float shared_alpha;
    __shared__ float shared_beta;
    
    // Thread indices
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Map block to (batch, head, token)
    int q_idx = blockIdx.x;
    int token_idx = q_idx % seq_len;
    int head_idx = (q_idx / seq_len) % num_heads;
    int batch_idx = q_idx / (seq_len * num_heads);
    
    // Calculate global memory offsets
    int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                      (head_idx * seq_len * head_dim);
    
    const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
    scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

    // Load Q into registers ONCE at kernel start
    float q_reg = static_cast<float>(q_vec[tid]);
    // Intialize O
    float o_reg = 0.0f;
    
    
    // Initialize online softmax state (shared across all threads)
    if (tid == 0) {
        shared_m_i = -INFINITY;
        shared_l_i = 0.0f;
    }
    
    
    
    // Iterate over Key/Value blocks
    for (int j_block = 0; j_block < seq_len; j_block += TILE_SIZE) {
        int valid_tile_size = min(TILE_SIZE, seq_len - j_block);
        
        // Process each K/V pair in the tile
        for (int j = 0; j < valid_tile_size; ++j) {
            int k_idx = j_block + j;
            const scalar_t* k_vec = K + base_offset + (k_idx * head_dim);
            const scalar_t* v_vec = V + base_offset + (k_idx * head_dim);
            
            // ===== PARALLEL DOT PRODUCT =====
            // Each thread computes partial dot product over its assigned dimensions
            float partial_score = q_reg * static_cast<float>(k_vec[tid]);
            
            // Reduce to get complete score
            float score = block_reduce_sum(partial_score, shared_mem);
            __syncthreads();
            
            // Thread 0 broadcasts the score and performs softmax update
            if (tid == 0) {
                score *= scale;
                
                // Online softmax update
                float m_prev = shared_m_i;
                shared_m_i = fmaxf(shared_m_i, score);
                
                shared_alpha = expf(m_prev - shared_m_i);
                shared_beta = expf(score - shared_m_i);
                
                shared_l_i = (shared_l_i * shared_alpha) + shared_beta;
            }
            __syncthreads();
            
            // All threads read the broadcast values
            float alpha = shared_alpha;
            float beta = shared_beta;
            
            // ===== PARALLEL OUTPUT UPDATE =====
            // Each thread updates its assigned dimensions
            float v_val = static_cast<float>(v_vec[tid]);
                
            o_reg = o_reg * alpha + v_val * beta;
                
        }
    }
    
    // ===== PARALLEL FINAL NORMALIZATION =====
    o_vec[tid] = static_cast<scalar_t>(o_reg / shared_l_i);
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

// Required: Main function that will be called from Python
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  // 1. Setup Output Tensor
  auto O = torch::empty_like(Q);

  // 2. Extract Dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  // 3. Configure Kernel Launch Parameters
  // Grid: One block per query token (Total threads = B * H * L)
  // Block: 1 thread (This is a simplified naive kernel)
  int total_threads = batch_size * num_heads * seq_len;
  dim3 blocks(total_threads);
  dim3 threads(head_dim);

  // 4. Dispatch and Launch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "optimized_flash_attention_kernel", ([&] {
        optimized_flash_attention_kernel<scalar_t><<<blocks, threads>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
            num_heads, seq_len, head_dim, scale);
      }));

  // 5. Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  // 6. Synchronize to ensure kernel completion
  // (Optional for standard PyTorch usage as generic stream syncs automatically,
  // but good for strict benchmarking boundaries)
  cudaDeviceSynchronize();

  return O;
}

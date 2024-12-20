#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "activation_func.h"

// CUDA kernel for custom attention forward pass
template <int activation_type, typename scalar_t>
__global__ void attention_mlp_forward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ attention_logits,
    int B, int T1, int T2, int C
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int c = threadIdx.x;

    __shared__ scalar_t shared_mem[1024];

    if (b >= B || i >= T1 || c >= C) return;

    scalar_t bias_val = bias[c];

    scalar_t sum = 0.0;
    scalar_t normalization = 0.0;
    for (int j = 0; j < T2; j++) {
        scalar_t q_val = Q[b * T1 * C + i * C + c];
        scalar_t k_val = K[b * T2 * C + j * C + c];
        scalar_t mask_val = mask[b * T1 * T2 + i * T2 + j];

        // Element-wise product plus bias.
        scalar_t logit = q_val * k_val + bias_val;
        // Activation function.
        scalar_t prob = activation_func<activation_type, scalar_t>(logit);
        // Reduce channel dimension taking mask into account.
        sum += prob * mask_val;
        normalization += mask_val;

        // Reduce across threads within the block for attention_logits
        scalar_t product = q_val * k_val * mask_val;

        int thread_idx = threadIdx.x;
        shared_mem[thread_idx] = product;
        __syncthreads();

        // Parallel reduction within the block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread_idx < stride) {
                shared_mem[thread_idx] += shared_mem[thread_idx + stride];
            }
            __syncthreads();
        }

        // Only thread 0 writes the reduced sum to attention_logits
        if (thread_idx == 0) {
            atomicAdd(&attention_logits[b * T1 * T2 + i * T2 + j], shared_mem[0]);
        }
    }
    output[b * T1 * C + i * C + c] = sum / normalization;
}

// CUDA kernel for custom attention backward pass
template <int activation_type, typename scalar_t>
__global__ void attention_mlp_backward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ mask,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ grad_attention_logits,
    scalar_t* __restrict__ grad_Q,
    scalar_t* __restrict__ grad_K,
    scalar_t* __restrict__ grad_bias,
    int B, int T1, int T2, int C
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int c = threadIdx.x;

    if (b >= B || i >= T1 || c >= C) return;

    scalar_t normalization = 0.0;
    for (int j = 0; j < T2; j++) {
        scalar_t mask_val = mask[b * T1 * T2 + i * T2 + j];
        normalization += mask_val;
    }
    scalar_t grad_output_val = grad_output[b * T1 * C + i * C + c] / normalization;

    scalar_t bias_val = bias[c];

    for (int j = 0; j < T2; j++) {
        scalar_t q_val = Q[b * T1 * C + i * C + c];
        scalar_t k_val = K[b * T2 * C + j * C + c];
        scalar_t mask_val = mask[b * T1 * T2 + i * T2 + j];

        // Recompute logit from forward pass
        scalar_t logit = q_val * k_val + bias_val;

        // Gradient of activation function
        scalar_t grad_logit = grad_output_val * activation_derivative<activation_type, scalar_t>(logit) * mask_val;

        // Gradient of bias (accumulate across i dimension)
        atomicAdd(&grad_bias[c], grad_logit);

        // Gradient of Q (accumulate across j dimension)
        scalar_t grad_q_val = grad_logit * k_val;
        grad_q_val += grad_attention_logits[b * T1 * T2 + i * T2 + j] * k_val * mask_val;
        atomicAdd(&grad_Q[b * T1 * C + i * C + c], grad_q_val);

        // Gradient of K (accumulate across i dimension)
        scalar_t grad_k_val = grad_logit * q_val;
        grad_k_val += grad_attention_logits[b * T1 * T2 + i * T2 + j] * q_val * mask_val;
        atomicAdd(&grad_K[b * T2 * C + j * C + c], grad_k_val);
    }
}

// Wrapper function for calling the forward kernel
template <int activation_type>
std::vector<torch::Tensor> attention_mlp_forward_cuda_template(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias,
    torch::Tensor mask
) {
    int B = Q.size(0);
    int T1 = Q.size(1);
    int T2 = K.size(1);
    int C = Q.size(2);

    TORCH_CHECK(K.size(0) == B && K.size(2) == C, "K must have shape (B, T2, C)");
    TORCH_CHECK(bias.size(0) == C, "bias must have shape (C,)");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T1 && mask.size(2) == T2, "mask must have shape (B, T1, T2)");

    torch::Tensor output = torch::zeros({B, T1, C}, Q.options());
    torch::Tensor attention_logits = torch::zeros({B, T1, T2}, Q.options());

    dim3 dimGrid(B, T1);
    dim3 dimBlock(C);

    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_mlp_forward_kernel", ([&] {
        attention_mlp_forward_kernel<activation_type, scalar_t><<<dimGrid, dimBlock>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            attention_logits.data_ptr<scalar_t>(),
            B, T1, T2, C
        );
    }));

    return {output, attention_logits};
}

// Call the appropiate template function based on the activation type.
std::vector<torch::Tensor> attention_mlp_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias,
    torch::Tensor mask,
    int activation_type
) {
    // Check tensor contiguity
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

    if (activation_type == 0) {
        return attention_mlp_forward_cuda_template<0>(Q, K, bias, mask);
    } else if (activation_type == 1) {
        return attention_mlp_forward_cuda_template<1>(Q, K, bias, mask);
    } else if (activation_type == 2) {
        return attention_mlp_forward_cuda_template<2>(Q, K, bias, mask);
    } else {
        // Handle invalid activation type (maybe raise an error)
        return attention_mlp_forward_cuda_template<0>(Q, K, bias, mask);
    }
}

// Wrapper function for calling the backward kernel
template <int activation_type>
std::vector<torch::Tensor> attention_mlp_backward_cuda_template(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias,
    torch::Tensor mask,
    torch::Tensor grad_output,
    torch::Tensor grad_attention_logits
) {
    int B = Q.size(0);
    int T1 = Q.size(1);
    int T2 = K.size(1);
    int C = Q.size(2);

    TORCH_CHECK(K.size(0) == B && K.size(2) == C, "K must have shape (B, T2, C)");
    TORCH_CHECK(bias.size(0) == C, "bias must have shape (C,)");
    TORCH_CHECK(mask.size(0) == B && mask.size(1) == T1 && mask.size(2) == T2, "mask must have shape (B, T1, T2)");

    torch::Tensor grad_Q = torch::zeros_like(Q);
    torch::Tensor grad_K = torch::zeros_like(K);
    torch::Tensor grad_bias = torch::zeros_like(bias);

    dim3 dimGrid(B, T1);
    dim3 dimBlock(C);

    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_mlp_backward_kernel", ([&] {
        attention_mlp_backward_kernel<activation_type, scalar_t><<<dimGrid, dimBlock>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_attention_logits.data_ptr<scalar_t>(),
            grad_Q.data_ptr<scalar_t>(),
            grad_K.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            B, T1, T2, C
        );
    }));

    return {grad_Q, grad_K, grad_bias};
}

// Call the appropiate template function based on the activation type.
std::vector<torch::Tensor> attention_mlp_backward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias,
    torch::Tensor mask,
    torch::Tensor grad_output,
    torch::Tensor grad_attention_logits,
    int activation_type
) {
    // Check tensor contiguity
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_attention_logits.is_contiguous(), "grad_attention_logits must be contiguous");

    if (activation_type == 0) {
        return attention_mlp_backward_cuda_template<0>(Q, K, bias, mask, grad_output, grad_attention_logits);
    } else if (activation_type == 1) {
        return attention_mlp_backward_cuda_template<1>(Q, K, bias, mask, grad_output, grad_attention_logits);
    } else if (activation_type == 2) {
        return attention_mlp_backward_cuda_template<2>(Q, K, bias, mask, grad_output, grad_attention_logits);
    } else {
        // Handle invalid activation type (maybe raise an error)
        return attention_mlp_backward_cuda_template<0>(Q, K, bias, mask, grad_output, grad_attention_logits);
    }
}

// PYBIND11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_mlp_forward_cuda", &attention_mlp_forward_cuda, "Fused MLP Attention Forward (CUDA)");
    m.def("attention_mlp_backward_cuda", &attention_mlp_backward_cuda, "Fused MLP Attention Backward (CUDA)");
}
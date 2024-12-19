#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for custom attention forward pass
template <typename scalar_t>
__global__ void attention_mlp_forward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int B, int T, int C,
    int bias_size
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int c = threadIdx.x;

    if (b >= B || i >= T || c >= C) return;

    scalar_t sum = 0.0;
    for (int j = 0; j < T; j++) {
        scalar_t q_val = Q[b * T * C + i * C + c];
        scalar_t k_val = K[b * T * C + j * C + c];
        scalar_t bias_val = bias[j * bias_size + c];

        // Element-wise product plus bias.
        scalar_t logit = q_val * k_val + bias_val;
        // Sigmoid.
        // TODO optionally allow ReLU and Tahn.
        scalar_t prob = 1.0 / (1.0 + exp(-logit));
        // Reduce channel dimension.
        sum += prob;
    }
    output[b * T * C + i * C + c] = sum / T;
}

// CUDA kernel for custom attention backward pass
template <typename scalar_t>
__global__ void attention_mlp_backward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_Q,
    scalar_t* __restrict__ grad_K,
    scalar_t* __restrict__ grad_bias,
    int B, int T, int C,
    int bias_size
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int c = threadIdx.x;

    if (b >= B || i >= T || c >= C) return;

    scalar_t grad_out_val = grad_output[b * T * C + i * C + c] / T;

    for (int j = 0; j < T; j++) {
        scalar_t q_val = Q[b * T * C + i * C + c];
        scalar_t k_val = K[b * T * C + j * C + c];
        scalar_t bias_val = bias[j * bias_size + c];

        // Recompute logit and prob from forward pass
        scalar_t logit = q_val * k_val + bias_val;
        scalar_t prob = 1.0 / (1.0 + exp(-logit));

        // Gradient of sigmoid
        scalar_t grad_logit = grad_out_val * prob * (1.0 - prob);

        // Gradient of bias (accumulate across i dimension)
        atomicAdd(&grad_bias[j * bias_size + c], grad_logit);

        // Gradient of Q (accumulate across j dimension)
        atomicAdd(&grad_Q[b * T * C + i * C + c], grad_logit * k_val);

        // Gradient of K (accumulate across i dimension)
        atomicAdd(&grad_K[b * T * C + j * C + c], grad_logit * q_val);
    }
}

// Wrapper function for calling the forward kernel
torch::Tensor attention_mlp_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias
) {
    int B = Q.size(0);
    int T = Q.size(1);
    int C = Q.size(2);
    int bias_size = bias.size(1);

    torch::Tensor output = torch::zeros({B, T, C}, Q.options());

    dim3 dimGrid(B, T);
    dim3 dimBlock(C);

    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_mlp_forward_kernel", ([&] {
        attention_mlp_forward_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, T, C, bias_size
        );
    }));

    return output;
}

// Wrapper function for calling the backward kernel
std::vector<torch::Tensor> attention_mlp_backward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor bias,
    torch::Tensor grad_output
) {
    int B = Q.size(0);
    int T = Q.size(1);
    int C = Q.size(2);
    int bias_size = bias.size(1);

    torch::Tensor grad_Q = torch::zeros_like(Q);
    torch::Tensor grad_K = torch::zeros_like(K);
    torch::Tensor grad_bias = torch::zeros_like(bias);

    dim3 dimGrid(B, T);
    dim3 dimBlock(C);

    AT_DISPATCH_FLOATING_TYPES(Q.type(), "attention_mlp_backward_kernel", ([&] {
        attention_mlp_backward_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_Q.data_ptr<scalar_t>(),
            grad_K.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            B, T, C, bias_size
        );
    }));

    return {grad_Q, grad_K, grad_bias};
}

// PYBIND11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_mlp_forward_cuda", &attention_mlp_forward_cuda, "Fused MLP Attention Forward (CUDA)");
    m.def("attention_mlp_backward_cuda", &attention_mlp_backward_cuda, "Fused MLP Attention Backward (CUDA)");
}
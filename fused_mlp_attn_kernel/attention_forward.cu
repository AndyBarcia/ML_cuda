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

// Wrapper function for calling the kernel
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

// PYBIND11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_mlp_forward_cuda", &attention_mlp_forward_cuda, "Fused MLP Attention Forward (CUDA)");
}
import torch
import torch.nn.functional as F
from torch.autograd import Function
import attention_mlp 

class AttentionMLP(Function):
    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def forward(ctx, Q, K, bias):
        # Save for backward
        ctx.save_for_backward(Q, K, bias)
        ctx.bias_shape = bias.shape

        # einsum('b i c, b j c -> b i j c', Q, K) + bias
        attention_logits = torch.einsum('bic,bjc->bijc', Q, K) + bias

        # F.sigmoid(attention_logits).mean(dim=-2)
        attention_probs = F.sigmoid(attention_logits)
        output = attention_probs.mean(dim=-2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        Q, K, bias = ctx.saved_tensors
        bias_shape = ctx.bias_shape

        # Recompute forward pass intermediates for efficiency
        attention_logits = torch.einsum('bic,bjc->bijc', Q, K) + bias
        attention_probs = F.sigmoid(attention_logits)

        # Backward of mean(dim=-2)
        grad_attention_probs = grad_output.unsqueeze(-2) / Q.shape[-2]  # Divide by T

        # Backward of sigmoid
        grad_attention_logits = grad_attention_probs * attention_probs * (1 - attention_probs)

        # Backward of bias (sum over i and j dimensions)
        grad_bias = grad_attention_logits.flatten(0,-len(bias_shape)-1).sum(dim=0).reshape(bias_shape)

        # Backward of einsum
        grad_Q = torch.einsum('bjc,bijc->bic', K, grad_attention_logits)
        grad_K = torch.einsum('bic,bijc->bjc', Q, grad_attention_logits)

        return grad_Q, grad_K, grad_bias

class AttentionMLP_CUDA(Function):
    @staticmethod
    def forward(ctx, Q, K, bias):
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, bias)
        ctx.bias_shape = bias.shape

        # Call the CUDA kernel for the forward pass
        output = attention_mlp.attention_mlp_forward_cuda(Q, K, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Placeholder for backward pass implementation
        # Retrieve saved tensors
        Q, K, bias = ctx.saved_tensors
        bias_shape = ctx.bias_shape

        # Allocate gradient tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_bias = torch.zeros(bias_shape, dtype=bias.dtype, device=bias.device)

        return grad_Q, grad_K, grad_bias

def profile_implementation(name, function, Q, K, bias):
    print(f"\nProfiling {name} Implementation:")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,  # Enables memory tracking
    ) as prof:
        output = function(Q, K, bias)
        output.sum().backward()

    # Print the results
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    return output

# Example usage:
if __name__ == "__main__":
    # Sample data
    B, T, C = 32, 128, 256  # Batch, Time, Channels
    bias_size = 4
    Q = torch.randn(B, T, C, device='cuda', requires_grad=True)
    K = torch.randn(B, T, C, device='cuda', requires_grad=True)
    bias = torch.randn(T, C, device='cuda', requires_grad=True)

    # Perform the CUDA forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_cuda = profile_implementation("CUDA", AttentionMLP_CUDA.apply, Q, K, bias)
    cuda_allocated = torch.cuda.memory_allocated()
    cuda_reserved = torch.cuda.memory_reserved()
    print(f"CUDA Implementation: Allocated={cuda_allocated}, Reserved={cuda_reserved}")
    print("Output CUDA shape:", output_cuda.shape)

    Q_grad_cuda, K_grad_cuda, bias_grad_cuda = Q.grad, K.grad, bias.grad
    # Reset gradients for automatic differentiation
    Q.grad, K.grad, bias.grad = None, None, None

    # Perform the PyTorch forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_pytorch = profile_implementation("PyTorch", AttentionMLP.apply, Q, K, bias)
    pytorch_allocated = torch.cuda.memory_allocated()
    pytorch_reserved = torch.cuda.memory_reserved()
    print(f"PyTorch Implementation: Allocated={pytorch_allocated}, Reserved={pytorch_reserved}")
    print("Output PyTorch shape:", output_pytorch.shape)

    Q_grad_pytorch, K_grad_pytorch, bias_grad_pytorch = Q.grad, K.grad, bias.grad
    # Reset gradients for automatic differentiation
    Q.grad, K.grad, bias.grad = None, None, None

    print("Output close:", torch.allclose(output_cuda, output_pytorch))
    print("Q grads close:", torch.allclose(Q_grad_cuda, Q_grad_pytorch))
    print("K grads close:", torch.allclose(K_grad_cuda, K_grad_pytorch))
    print("Bias grads close:", torch.allclose(bias_grad_cuda, bias_grad_pytorch))

    print("(output_cuda-output_pytorch).abs().max()", (output_cuda-output_pytorch).abs().max())
    print("(Q_grad_cuda-Q_grad_pytorch).abs().max()", (Q_grad_cuda-Q_grad_pytorch).abs().max())
    print("(K_grad_cuda-K_grad_pytorch).abs().max()", (K_grad_cuda-K_grad_pytorch).abs().max())
    print("(bias_grad_cuda-bias_grad_pytorch).abs().max()", (bias_grad_cuda-bias_grad_pytorch).abs().max())
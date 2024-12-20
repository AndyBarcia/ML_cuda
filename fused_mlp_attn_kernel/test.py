import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from attention_mlp.attention import AttentionMLP_CUDA, ActivationType

class AttentionMLP_Autograd(nn.Module):
    def forward(self, Q, K, bias, mask):
        # einsum('b i c, b j c -> b i j c', Q, K) + bias
        attention_logits = torch.einsum('bic,bjc->bijc', Q, K) + bias

        # F.sigmoid(attention_logits).mean(dim=-2)
        attention_probs = F.sigmoid(attention_logits)
        # Masked mean
        masked_attention_probs = attention_probs * mask.unsqueeze(-1)
        output = masked_attention_probs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)

        # Normal dot-product between all pairs of queries and keys
        attention_logits = torch.einsum('bic,bjc,bij->bij', Q, K, mask)

        return output, attention_logits

class AttentionMLP_PyTorch(Function):
    @staticmethod
    def forward(ctx, Q, K, bias, mask):
        # Save for backward
        ctx.save_for_backward(Q, K, bias, mask)
        ctx.bias_shape = bias.shape

        # einsum('b i c, b j c -> b i j c', Q, K) + bias
        attention_logits = torch.einsum('bic,bjc->bijc', Q, K) + bias

        # F.sigmoid(attention_logits).mean(dim=-2)
        attention_probs = F.sigmoid(attention_logits)
        # Masked mean
        masked_attention_probs = attention_probs * mask.unsqueeze(-1)
        output = masked_attention_probs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)

        # Normal dot-product between all pairs of queries and keys
        attention_logits = torch.einsum('bic,bjc,bij->bij', Q, K, mask)

        return output, attention_logits

    @staticmethod
    def backward(ctx, grad_output, grad_attention_logits):
        # Retrieve saved tensors
        Q, K, bias, mask = ctx.saved_tensors
        bias_shape = ctx.bias_shape

        # Recompute forward pass intermediates
        attention_logits_with_bias = torch.einsum('bic,bjc->bijc', Q, K) + bias
        attention_probs = F.sigmoid(attention_logits_with_bias)

        # Backward of masked mean for output
        grad_masked_attention_probs = grad_output.unsqueeze(-2) / mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        grad_attention_probs = grad_masked_attention_probs * mask.unsqueeze(-1)

        # Backward of sigmoid for output
        grad_attention_logits_from_output = grad_attention_probs * attention_probs * (1 - attention_probs)

        # Backward of bias (sum over i dimension for output)
        grad_bias = grad_attention_logits_from_output.sum(dim=-3)

        # Backward of einsum for output
        grad_Q_from_output = torch.einsum('bjc,bijc->bic', K, grad_attention_logits_from_output)
        grad_K_from_output = torch.einsum('bic,bijc->bjc', Q, grad_attention_logits_from_output)

        # Backward of einsum for attention_logits (masked)
        grad_Q_from_logits = torch.einsum('bjc,bij->bic', K, grad_attention_logits * mask)
        grad_K_from_logits = torch.einsum('bic,bij->bjc', Q, grad_attention_logits * mask)

        # Combine gradients
        grad_Q = grad_Q_from_output + grad_Q_from_logits
        grad_K = grad_K_from_output + grad_K_from_logits

        return grad_Q, grad_K, grad_bias, None  # No gradient for the mask

def profile_implementation(name, function, Q, K, bias, mask, *args):
    print(f"\nProfiling {name} Implementation:")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,  # Enables memory tracking
    ) as prof:
        output, attn_logits = function(Q, K, bias, mask, *args)
        if attn_logits is not None:
            (output.sum() + attn_logits.sum()).backward()
        else:
            output.sum().backward()

    # Print the results
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    return output, attn_logits

# Example usage:
if __name__ == "__main__":
    # Sample data
    B, T1, T2, C = 32, 32, 24, 512  # Batch, Time, Channels
    Q = torch.randn(B, T1, C, device='cuda', requires_grad=True, dtype=torch.float64)
    K = torch.randn(B, T2, C, device='cuda', requires_grad=True, dtype=torch.float64)
    bias = torch.randn(C, device='cuda', requires_grad=True, dtype=torch.float64)
    mask = torch.randn(B, T1, T2, device='cuda', dtype=torch.float64)

    # Perform the CUDA forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_cuda, attn_cuda = profile_implementation("CUDA", AttentionMLP_CUDA.apply, Q, K, bias, mask, ActivationType.SIGMOID.to_int())
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
    output_pytorch, attn_pytorch = profile_implementation("PyTorch", AttentionMLP_PyTorch.apply, Q, K, bias, mask)
    pytorch_allocated = torch.cuda.memory_allocated()
    pytorch_reserved = torch.cuda.memory_reserved()
    print(f"PyTorch Implementation: Allocated={pytorch_allocated}, Reserved={pytorch_reserved}")
    print("Output PyTorch shape:", output_pytorch.shape)

    Q_grad_pytorch, K_grad_pytorch, bias_grad_pytorch = Q.grad, K.grad, bias.grad
    # Reset gradients for automatic differentiation
    Q.grad, K.grad, bias.grad = None, None, None

    print("Output close:", torch.allclose(output_cuda, output_pytorch))
    print("Attn close:", torch.allclose(attn_cuda, attn_pytorch))
    print("Q grads close:", torch.allclose(Q_grad_cuda, Q_grad_pytorch))
    print("K grads close:", torch.allclose(K_grad_cuda, K_grad_pytorch))
    print("Bias grads close:", torch.allclose(bias_grad_cuda, bias_grad_pytorch))

    print("(attn_cuda-attn_pytorch).abs().max()", (attn_cuda-attn_pytorch).abs().max())
    print("(output_cuda-output_pytorch).abs().max()", (output_cuda-output_pytorch).abs().max())
    print("(Q_grad_cuda-Q_grad_pytorch).abs().max()", (Q_grad_cuda-Q_grad_pytorch).abs().max())
    print("(K_grad_cuda-K_grad_pytorch).abs().max()", (K_grad_cuda-K_grad_pytorch).abs().max())
    print("(bias_grad_cuda-bias_grad_pytorch).abs().max()", (bias_grad_cuda-bias_grad_pytorch).abs().max())

    # Perform the PyTorch Autograd backward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_pytorch_auto, attn_pytorch_auto = profile_implementation("PyTorch Auto", AttentionMLP_Autograd(), Q, K, bias, mask)
    pytorch_allocated = torch.cuda.memory_allocated()
    pytorch_reserved = torch.cuda.memory_reserved()
    print(f"PyTorch AutoGrad Implementation: Allocated={pytorch_allocated}, Reserved={pytorch_reserved}")
    print("Output PyTorch AutoGrad shape:", output_pytorch_auto.shape)

    Q_grad_pytorch_auto, K_grad_pytorch_auto, bias_grad_pytorch_auto = Q.grad, K.grad, bias.grad
    # Reset gradients for automatic differentiation
    Q.grad, K.grad, bias.grad = None, None, None

    print("Output close:", torch.allclose(output_cuda, output_pytorch_auto))
    print("Attn close:", torch.allclose(attn_cuda, attn_pytorch_auto))
    print("Q grads close:", torch.allclose(Q_grad_cuda, Q_grad_pytorch_auto))
    print("K grads close:", torch.allclose(K_grad_cuda, K_grad_pytorch_auto))
    print("Bias grads close:", torch.allclose(bias_grad_cuda, bias_grad_pytorch_auto))

    print("(attn_cuda-attn_pytorch_auto).abs().max()", (attn_cuda-attn_pytorch_auto).abs().max())
    print("(output_cuda-output_pytorch_auto).abs().max()", (output_cuda-output_pytorch_auto).abs().max())
    print("(Q_grad_cuda-Q_grad_pytorch_auto).abs().max()", (Q_grad_cuda-Q_grad_pytorch_auto).abs().max())
    print("(K_grad_cuda-K_grad_pytorch_auto).abs().max()", (K_grad_cuda-K_grad_pytorch_auto).abs().max())
    print("(bias_grad_cuda-bias_grad_pytorch_auto).abs().max()", (bias_grad_cuda-bias_grad_pytorch_auto).abs().max())
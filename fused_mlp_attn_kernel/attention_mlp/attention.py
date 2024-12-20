import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import attention_mlp_cuda 
import enum


class AttentionMLP_CUDA(Function):
    @staticmethod
    def forward(ctx, Q, K, bias, mask, activation_type):
        # Check tensor contiguity and data type
        assert Q.is_contiguous() and K.is_contiguous() and bias.is_contiguous() and mask.is_contiguous(), "Tensors must be contiguous"
        assert Q.dtype == K.dtype == bias.dtype, "Tensors must have the same data type"

        # Check tensor shapes
        B, T1, C = Q.shape
        _, T2, _ = K.shape
        assert mask.shape == (B, T1, T2), f"Mask shape must be (B, T, T), got {mask.shape}"
        assert bias.shape[0] == C, f"Bias shape must be (C,), got {bias.shape}"

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, bias, mask)
        ctx.activation_type = activation_type

        # Call the CUDA kernel for the forward pass
        output, attention_logits = attention_mlp_cuda.attention_mlp_forward_cuda(Q, K, bias, mask, activation_type)

        return output, attention_logits

    @staticmethod
    def backward(ctx, grad_output, grad_attention_logits):
        # Retrieve saved tensors
        Q, K, bias, mask = ctx.saved_tensors
        activation_type = ctx.activation_type

        # Check tensor contiguity and data type
        #assert grad_output.is_contiguous() and grad_attention_logits.is_contiguous(), "Gradients must be contiguous"
        assert grad_output.dtype == grad_attention_logits.dtype == Q.dtype, "Gradients and Q must have the same data type"

        # Call the CUDA kernel for the backward pass
        grad_Q, grad_K, grad_bias = attention_mlp_cuda.attention_mlp_backward_cuda(
            Q, 
            K, 
            bias, 
            mask, 
            grad_output.contiguous(), 
            grad_attention_logits.contiguous(), 
            activation_type
        )

        return grad_Q, grad_K, grad_bias, None, None  # No gradients for mask and activation_type

class ActivationType(enum.Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TAHN = "tahn"

    def to_int(self):
        if self == ActivationType.SIGMOID:
            return 0
        elif self == ActivationType.RELU:
            return 1
        elif self == ActivationType.TAHN:
            return 2
        return 0

class AttentionMLP(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        activation_type: ActivationType = ActivationType.SIGMOID,
        dropout=0.1
    ):
        super(AttentionMLP, self).__init__()

        self.activation_type = activation_type
        self.activation_type_i = activation_type.to_int()

        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.bias = nn.Parameter(torch.randn(embedding_dim))

        self.W_v = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, q, k, mask):
        """
        Args:
            Q: Tensor of shape (B, T, C)
            K: Tensor of shape (B, T, C)
            mask: Tensor of shape (B, T, T)
        Returns:
            output: Tensor of shape (B, T, C)
            attention_logits: Tensor of shape (B, T, T)
        """

        Q = self.W_q(q) # (B, T_x, D)
        K = self.W_k(k) # (B, T_y, D)

        output, attention_logits = AttentionMLP_CUDA.apply(
            Q, 
            K, 
            self.bias, 
            mask, 
            self.activation_type_i
        ) # (B, T_x, D), (B, T_x, T_y)

        output = self.W_v(output) # (B, T_x, D)

        # Call the custom CUDA autograd function
        return output, attention_logits
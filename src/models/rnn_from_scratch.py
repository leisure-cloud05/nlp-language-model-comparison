"""Single-layer Vanilla RNN implemented using PyTorch tensor operations."""

import torch
from torch import nn


class RNNFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Input and hidden sizes must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Match nn.RNN's parameter shapes and conventions.
        self.weight_ih_l0 = nn.Parameter(
            torch.empty(hidden_size, input_size)
        )
        self.weight_hh_l0 = nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih_l0 = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Match PyTorch RNN's initialization convention.
        bound = self.hidden_size ** -0.5

        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, inputs, hidden=None):
        """
        inputs: [batch, seq_len, input_size]
        hidden: optional [1, batch, hidden_size]

        Returns:
            outputs: [batch, seq_len, hidden_size]
            final_hidden: [1, batch, hidden_size]
        """
        if inputs.ndim != 3:
            raise ValueError("Expected [batch, seq_len, input_size].")

        batch_size, seq_len, input_size = inputs.shape

        if input_size != self.input_size:
            raise ValueError("Unexpected input feature dimension.")
        if seq_len == 0:
            raise ValueError("Sequence length must be positive.")

        if hidden is None:
            h = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            if hidden.shape != (1, batch_size, self.hidden_size):
                raise ValueError("Invalid initial hidden-state shape.")
            h = hidden[0]

        outputs = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]

            h = torch.tanh(
                torch.nn.functional.linear(
                    x_t, self.weight_ih_l0, self.bias_ih_l0
                )
                + torch.nn.functional.linear(
                    h, self.weight_hh_l0, self.bias_hh_l0
                )
            )

            outputs.append(h)

        return torch.stack(outputs, dim=1), h.unsqueeze(0)

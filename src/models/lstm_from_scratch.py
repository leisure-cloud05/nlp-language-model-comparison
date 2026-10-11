"""Single-layer LSTM implemented from scratch using PyTorch tensors."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Input and hidden sizes must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Concatenated parameters for four LSTM gates: i, f, g, o.
        gate_size = 4 * hidden_size

        self.weight_ih_l0 = nn.Parameter(
            torch.empty(gate_size, input_size)
        )
        self.weight_hh_l0 = nn.Parameter(
            torch.empty(gate_size, hidden_size)
        )
        self.bias_ih_l0 = nn.Parameter(torch.empty(gate_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(gate_size))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, inputs, state=None):
        """
        Args:
            inputs: [batch, seq_len, input_size]
            state: optional (h0, c0), each [1, batch, hidden_size]

        Returns:
            outputs: [batch, seq_len, hidden_size]
            final_state: (h_n, c_n), each [1, batch, hidden_size]
        """
        if inputs.ndim != 3:
            raise ValueError("Expected [batch, seq_len, input_size].")

        batch_size, seq_len, input_size = inputs.shape

        if input_size != self.input_size:
            raise ValueError("Unexpected input feature dimension.")
        if seq_len == 0:
            raise ValueError("Sequence length must be positive.")

        if state is None:
            h = inputs.new_zeros(batch_size, self.hidden_size)
            c = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            if not isinstance(state, (tuple, list)) or len(state) != 2:
                raise ValueError("State must be a tuple (h0, c0).")

            h0, c0 = state
            expected_shape = (1, batch_size, self.hidden_size)

            if h0.shape != expected_shape or c0.shape != expected_shape:
                raise ValueError("Invalid initial state shape.")

            h = h0[0]
            c = c0[0]

        outputs = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]

            gates = (
                F.linear(x_t, self.weight_ih_l0, self.bias_ih_l0)
                + F.linear(h, self.weight_hh_l0, self.bias_hh_l0)
            )

            # PyTorch gate order: input, forget, candidate, output.
            i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)

            i = torch.sigmoid(i_raw)
            f = torch.sigmoid(f_raw)
            g = torch.tanh(g_raw)
            o = torch.sigmoid(o_raw)

            # Cell and hidden-state updates.
            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        final_state = (h.unsqueeze(0), c.unsqueeze(0))

        return outputs, final_state

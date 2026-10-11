"""Compare a manual RNN implementation with PyTorch nn.RNN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.models.rnn_from_scratch import RNNFromScratch


def main():
    torch.manual_seed(42)

    input_size = 8
    hidden_size = 16
    batch_size = 3
    seq_len = 5

    reference = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        batch_first=True,
    )

    manual = RNNFromScratch(input_size, hidden_size)

    # Copy identical weights and biases.
    manual.load_state_dict(reference.state_dict())

    inputs = torch.randn(batch_size, seq_len, input_size)
    initial_hidden = torch.randn(1, batch_size, hidden_size)

    # Separate tensors for gradient comparison.
    x_ref = inputs.clone().requires_grad_(True)
    x_manual = inputs.clone().requires_grad_(True)

    h_ref = initial_hidden.clone().requires_grad_(True)
    h_manual = initial_hidden.clone().requires_grad_(True)

    ref_outputs, ref_hidden = reference(x_ref, h_ref)
    manual_outputs, manual_hidden = manual(x_manual, h_manual)

    torch.testing.assert_close(
        manual_outputs, ref_outputs, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        manual_hidden, ref_hidden, rtol=1e-5, atol=1e-6
    )

    print("Forward outputs and final hidden state: PASS")

    # Use the same scalar objective to compare gradients.
    ref_loss = ref_outputs.square().sum() + ref_hidden.square().sum()
    manual_loss = (
        manual_outputs.square().sum() + manual_hidden.square().sum()
    )

    ref_loss.backward()
    manual_loss.backward()

    torch.testing.assert_close(
        x_manual.grad, x_ref.grad, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        h_manual.grad, h_ref.grad, rtol=1e-4, atol=1e-5
    )

    for name, parameter in manual.named_parameters():
        reference_parameter = dict(reference.named_parameters())[name]
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=1e-4,
            atol=1e-5,
        )

    print("Input, hidden-state and parameter gradients: PASS")

    # Also verify the zero-initial-state behavior.
    zero_ref, zero_ref_hidden = reference(inputs)
    zero_manual, zero_manual_hidden = manual(inputs)

    torch.testing.assert_close(
        zero_manual, zero_ref, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        zero_manual_hidden, zero_ref_hidden,
        rtol=1e-5, atol=1e-6
    )

    print("Default zero initial hidden state: PASS")
    print("\nAll from-scratch RNN tests passed!")


if __name__ == "__main__":
    main()

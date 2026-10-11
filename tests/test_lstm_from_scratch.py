
"""Compare the manual LSTM with PyTorch nn.LSTM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.models.lstm_from_scratch import LSTMFromScratch


def main():
    torch.manual_seed(42)

    input_size = 8
    hidden_size = 16
    batch_size = 3
    seq_len = 5

    reference = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
    )

    manual = LSTMFromScratch(input_size, hidden_size)

    # Copy identical reference parameters.
    manual.load_state_dict(reference.state_dict())

    inputs = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.randn(1, batch_size, hidden_size)
    c0 = torch.randn(1, batch_size, hidden_size)

    x_ref = inputs.clone().requires_grad_(True)
    x_manual = inputs.clone().requires_grad_(True)

    h_ref = h0.clone().requires_grad_(True)
    h_manual = h0.clone().requires_grad_(True)

    c_ref = c0.clone().requires_grad_(True)
    c_manual = c0.clone().requires_grad_(True)

    ref_outputs, (ref_h, ref_c) = reference(
        x_ref, (h_ref, c_ref)
    )

    manual_outputs, (manual_h, manual_c) = manual(
        x_manual, (h_manual, c_manual)
    )

    torch.testing.assert_close(
        manual_outputs, ref_outputs,
        rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        manual_h, ref_h,
        rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        manual_c, ref_c,
        rtol=1e-5, atol=1e-6
    )

    print("Forward outputs and LSTM states: PASS")

    ref_loss = (
        ref_outputs.square().sum()
        + ref_h.square().sum()
        + ref_c.square().sum()
    )

    manual_loss = (
        manual_outputs.square().sum()
        + manual_h.square().sum()
        + manual_c.square().sum()
    )

    ref_loss.backward()
    manual_loss.backward()

    for manual_grad, ref_grad in [
        (x_manual.grad, x_ref.grad),
        (h_manual.grad, h_ref.grad),
        (c_manual.grad, c_ref.grad),
    ]:
        torch.testing.assert_close(
            manual_grad, ref_grad,
            rtol=1e-4, atol=1e-5
        )

    ref_parameters = dict(reference.named_parameters())

    for name, parameter in manual.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            ref_parameters[name].grad,
            rtol=1e-4,
            atol=1e-5,
        )

    print("Input, state and parameter gradients: PASS")

    # Check default zero initial hidden and cell states.
    zero_ref, (zero_ref_h, zero_ref_c) = reference(inputs)
    zero_manual, (zero_manual_h, zero_manual_c) = manual(inputs)

    torch.testing.assert_close(
        zero_manual, zero_ref,
        rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        zero_manual_h, zero_ref_h,
        rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        zero_manual_c, zero_ref_c,
        rtol=1e-5, atol=1e-6
    )

    print("Default zero initial states: PASS")
    print("\nAll from-scratch LSTM tests passed!")


if __name__ == "__main__":
    main()

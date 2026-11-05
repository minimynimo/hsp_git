"""Asserts that models.networks.EALSTM (cuDNN form) reproduces the reference EA-LSTM
(14_LSTM/ealstm_regional_modeling/papercode/ealstm.py) exactly: outputs and gradients."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "14_LSTM" / "ealstm_regional_modeling"))
from models.networks import EALSTM  # noqa: E402
from papercode.ealstm import EALSTM as RefEALSTM  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D, S, H, B, T = 3, 27, 64, 16, 365
fast = EALSTM(D, S, H, initial_forget_bias=5, dropout=0.0).to(dev).double()
ref = RefEALSTM(D, S, H, initial_forget_bias=5).to(dev).double()
with torch.no_grad():
    for n in ["weight_ih", "weight_hh", "weight_sh", "bias", "bias_s"]:
        p = torch.randn_like(getattr(ref, n)) * 0.3
        getattr(ref, n).copy_(p)
        getattr(fast, n).copy_(p)
x_d = torch.randn(B, T, D, device=dev, dtype=torch.double)
x_s = torch.randn(B, S, device=dev, dtype=torch.double)
y_fast = fast(x_d, x_s)
h_ref, _ = ref(x_d, x_s)
y_ref = fast.fc(h_ref[:, -1, :])
err = (y_fast - y_ref).abs().max().item()
y_fast.sum().backward(); y_ref.sum().backward()
gerr = max((getattr(fast, n).grad - getattr(ref, n).grad).abs().max().item()
           for n in ["weight_ih", "weight_hh", "weight_sh", "bias", "bias_s"])
print(f"max |output diff| = {err:.2e}, max |grad diff| = {gerr:.2e}")
assert err < 1e-8 and gerr < 1e-6, "EA-LSTM mismatch"
print("CHECK PASSED")

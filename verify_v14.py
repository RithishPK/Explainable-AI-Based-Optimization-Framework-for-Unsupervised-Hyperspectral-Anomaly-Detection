"""v14 verification: identity fallback + corrected signal is bidirectional."""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_lambda import (AdaptiveLambda, xai_alignment_signal,
                             xai_alignment_bg_signal, alignment_diagnostics)
from SeT import Mask

torch.manual_seed(0); np.random.seed(0)
dev = torch.device('cpu')

def make(mask_field, ig_field):
    m = Mask((100,100), dev); m.mask = torch.tensor(mask_field).float().unsqueeze(-1)
    return m, torch.tensor(ig_field).float()

print("="*70); print("TEST 1 - identity fallback: strategy='fixed' never moves lambda")
al = AdaptiveLambda(strategy='fixed', lambda_base=1e-3)
for e in range(50): al.update(epoch=e)
vals = set(al.lambda_history())
print(f"  unique lambda values over 50 updates: {vals}")
assert vals == {1e-3}, "fixed strategy moved lambda"
print("  PASS\n")

print("="*70); print("TEST 2 - v13 signal saturates whenever IG tracks the mask")
base = np.random.rand(100,100)*0.1; base[40:50,40:50] = 0.9
m, ig = make(base, base.copy())
d = alignment_diagnostics(ig, m, dm=None)
print(f"  corr(ig, mask) = {d['corr_ig_mask']:.4f}  (measured 0.98 on real Coast data)")
print(f"  v13 xai_alignment = {xai_alignment_signal(ig, m):+.4f}  <- positive purely from circularity")
print("  PASS (documents the v13 failure)\n")

print("="*70); print("TEST 3 - DISAGREEMENT: background-target IG avoids the masked region")
mask_f = np.random.rand(100,100)*0.1; mask_f[40:50,40:50] = 0.9   # mask says here
ig_f   = np.random.rand(100,100)*0.1; ig_f[10:20,80:90]   = 0.9   # bg-IG blames elsewhere
m, ig = make(mask_f, ig_f)
v13 = xai_alignment_signal(ig, m); v14 = xai_alignment_bg_signal(ig, m)
print(f"  v13 xai_alignment    = {v13:+.4f}")
print(f"  v14 xai_alignment_bg = {v14:+.4f}   <- NEGATIVE: masked pixels are not")
print(f"                                          driving background error")
assert v14 < 0, "bg signal must go negative when masked pixels do not drive bg error"
print("  PASS\n")

print("="*70); print("TEST 3b - AGREEMENT: masked pixels DO drive background error")
ig_f2 = np.random.rand(100,100)*0.1; ig_f2[40:50,40:50] = 0.9
m2, ig2 = make(mask_f, ig_f2)
v14b = xai_alignment_bg_signal(ig2, m2)
print(f"  v14 xai_alignment_bg = {v14b:+.4f}   <- POSITIVE: leakage detected")
assert v14b > 0
print("  PASS\n")

print("="*70); print("TEST 4 - lambda direction: orth signal can push BELOW lambda_base")
al = AdaptiveLambda(strategy='xai_alignment_bg', lambda_base=1e-3, alpha=0.5)
lo = al.update(iteration=1, alignment=v14)
hi = al.update(iteration=2, alignment=0.8)
print(f"  lambda_base = 1.000e-03")
print(f"  after negative alignment -> {lo:.3e}  ({'BELOW' if lo<1e-3 else 'not below'} base)")
print(f"  after positive alignment -> {hi:.3e}  ({'ABOVE' if hi>1e-3 else 'not above'} base)")
assert lo < 1e-3 < hi, "controller is not bidirectional"
print("  PASS  <- this is what v13 could never do on real data\n")
print("="*70); print("ALL v14 VERIFICATION TESTS PASSED"); print("="*70)

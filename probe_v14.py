"""Real-data check: does background-targeted IG break the 0.98 correlation?"""
import os, sys, numpy as np, torch
from torch.optim import Adam
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import coast
from model import MSNet
import select_bands, utils, xai_core
from metric import roc_auc
from SeT import TotalLoss, Mask
from SeT.detect import detect
from SeT.train import train_model
from adaptive_lambda import (AdaptiveLambda, xai_alignment_signal,
                             xai_alignment_bg_signal, alignment_diagnostics)

torch.manual_seed(0); np.random.seed(0)
dev = torch.device('cpu')
data, gt = coast.get_data(); rows, cols, _ = data.shape
bi = select_bands.OPBS(data, 64)
xb = utils.ZScoreNorm().fit(data[:,:,bi]).transform(data[:,:,bi])
x = torch.from_numpy(xb).to(dev).float()
model = MSNet(num_layers=3, shape=(rows,cols,64)).to(dev).float()
loss = TotalLoss(1e-3, dev); mask = Mask((rows,cols), dev)
opt = Adam(model.parameters(), lr=1e-3)
al = AdaptiveLambda(strategy='xai_alignment_bg', lambda_base=1e-3, alpha=0.5)

print(f"{'iter':>4} {'AUC':>8} | {'corr TOTAL':>11} {'v13 align':>10} | "
      f"{'corr BG':>9} {'v14 align':>10} {'lambda':>10}")
print("-"*76)
for i in range(1, 4):
    train_model(x, model, loss, {'mask':mask}, 150, opt, verbose=False,
                adaptive_lambda=al, epoch_offset=(i-1)*150)
    dm = detect(x, model(x)); mask.update(dm.detach())
    m2 = mask.mask.squeeze(-1)

    ig_tot = xai_core.ig_spatial_saliency(model, x.detach(), steps=20)
    ig_bg  = xai_core.ig_spatial_saliency(model, x.detach(), steps=20,
                                          target_weight=(1.0-m2).detach())
    d_tot = alignment_diagnostics(ig_tot, mask, None)
    d_bg  = alignment_diagnostics(ig_bg,  mask, None)
    a13 = xai_alignment_signal(ig_tot, mask)
    a14 = xai_alignment_bg_signal(ig_bg, mask)
    lam = al.update(iteration=i, alignment=a14)
    _,_,auc = roc_auc(dm.cpu().detach().numpy(), gt)
    print(f"{i:>4} {auc:>8.4f} | {d_tot['corr_ig_mask']:>11.4f} {a13:>10.4f} | "
          f"{d_bg['corr_ig_mask']:>9.4f} {a14:>10.4f} {lam:>10.3e}")

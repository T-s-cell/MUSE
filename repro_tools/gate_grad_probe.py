"""One-off diagnostic: why does gate_loss.backward() give sigmoid'(logit) exactly?"""
import torch

logit = torch.nn.Parameter(torch.tensor(-0.0002))
torch.manual_seed(1)
c = torch.randn(32, 96, 7)
t = torch.randn(32, 96, 7)
tgt = torch.randn(32, 96, 7)
g = torch.sigmoid(logit)
gp = g * c.detach() + (1 - g) * t.detach()
loss = torch.mean((gp - tgt) ** 2)
loss.backward()
M = (2 * (gp - tgt) * (c - t)).mean()
print("minimal: grad =", logit.grad.item(), " M*sig' =", (M * 0.25).item())

import numpy as np  # noqa
from ts_benchmark.baselines.muse.muse import MUSE
from repro_tools.audit import _loss_parts

state = torch.load(
    "/dev_data/wlt/MUSE_logs/R1d/full_model_step3.pt",
    map_location="cpu", weights_only=True,
)
m = MUSE(
    seq_len=1440, horizon=96,
    checkpoint_path="/dev_data/wlt/dmmv_rebuild/mae/mae_visualize_vit_base.pth",
    periodicity=24, freq="H",
)
m.model = m._init_model()
m.model.load_state_dict(state, strict=True)
m.model.to("cuda")
m._series_dim = 7
batch = (
    torch.randn(4, 1440, 7, device="cuda"),
    torch.randn(4, 816, 7, device="cuda"),
    torch.randn(4, 1440, 4, device="cuda"),
    torch.randn(4, 816, 4, device="cuda"),
)
core, ptgt, channel, tp, cl, tl, gl = _loss_parts(m, batch)
print("gate_loss:", gl.item(), "requires_grad:", gl.requires_grad)

fn = gl.grad_fn
depth = 0
while fn is not None and depth < 15:
    print("  " * depth, type(fn).__name__)
    nxt = None
    vals = getattr(fn, "next_functions", None)
    if vals:
        for v, _ in vals:
            if v is not None:
                print("  " * (depth + 1), "->", type(v).__name__)
        nxt = vals[0][0]
    fn = nxt
    depth += 1

print("fusion_logit is_leaf:", core.fusion_logit.is_leaf,
      "requires_grad:", core.fusion_logit.requires_grad)
core.zero_grad()
gl.backward()
print("fusion_logit.grad after gl.backward():", core.fusion_logit.grad)

# does gate itself carry grad_fn? is the sigmoid in _loss_parts on the live param?
gate = torch.sigmoid(core.fusion_logit)
print("gate.grad_fn:", gate.grad_fn)
gp2 = gate * channel.detach() + (1 - gate) * tp.detach()
print("gp2 grad_fn:", gp2.grad_fn)
manual = torch.mean((gp2 - ptgt) ** 2)
core.zero_grad()
manual.backward()
print("manual-recompute grad:", core.fusion_logit.grad.item())

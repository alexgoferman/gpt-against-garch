"""Zero-shot Chronos: one-day-ahead predictive distributions, rolling, no training.

For every forecast day t and every industry, the model sees only returns up to t-1
(the last CONTEXT days) and samples NUM_SAMPLES values for day t.

Output: results/forecasts/<name>_<period>.npz with
    samples: float32 [n_days, 49, NUM_SAMPLES]   predictive samples (decimal returns)
    dates:   forecast dates (YYYY-MM-DD strings)
    industries: column names
This is the shared format: GARCH and random walk should save the same shape so that
metrics.py can score all four models the same way.

Example:
    python src/chronos_zero_shot.py --period test
    python src/chronos_zero_shot.py --model checkpoints/run-0/checkpoint-final --name chronos_ft
"""
import argparse, os, time
import numpy as np
import torch
from chronos import ChronosPipeline
from data_utils import load_french49, VAL_START, VAL_END, TEST_START

p = argparse.ArgumentParser()
p.add_argument("--table", default="value", choices=["value", "equal"])
p.add_argument("--model", default="amazon/chronos-t5-mini")   # or a fine-tuned checkpoint dir
p.add_argument("--name", default="chronos_zs")
p.add_argument("--period", default="test", choices=["val", "test"])
p.add_argument("--context", type=int, default=512)
p.add_argument("--num_samples", type=int, default=100)
p.add_argument("--batch", type=int, default=None,
               help="series per predict call; default 8 on CPU, 64 on GPU (each series uses ~1.2 GB in float32)")
p.add_argument("--max_days", type=int, default=None, help="debug: only first N days")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed)
df = load_french49(a.table)
start, end = (VAL_START, VAL_END) if a.period == "val" else (TEST_START, None)
days = df.loc[start:end].index
if a.max_days:
    days = days[: a.max_days]

# Device: NVIDIA GPU > Apple GPU (MPS, ~2.5x faster than CPU on an M4 Max) > CPU.
# bfloat16 is fast on CUDA but ~5x slower than float32 on a Mac CPU, so use float32 off CUDA.
cuda = torch.cuda.is_available()
device = "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
# Chronos copies every 512-day context num_samples times before the encoder, so memory grows fast with
# batch size. 8 was fastest on MPS (16 was slower); 64 on CUDA is an estimate, lower it if you hit OOM.
a.batch = a.batch or (64 if cuda else 8)
pipe = ChronosPipeline.from_pretrained(
    a.model, device_map=device, torch_dtype=torch.bfloat16 if cuda else torch.float32
)
print("device:", device, "batch:", a.batch, flush=True)

values = df.to_numpy(dtype=np.float32)          # [T, 49], NaN = missing
pos = {d: i for i, d in enumerate(df.index)}
n_ind = values.shape[1]
out = np.full((len(days), n_ind, a.num_samples), np.nan, dtype=np.float32)

# Build (day, industry) jobs; context = the CONTEXT rows strictly before the forecast day.
jobs = [(k, j) for k in range(len(days)) for j in range(n_ind)]
t0 = time.time()
for b in range(0, len(jobs), a.batch):
    chunk = jobs[b : b + a.batch]
    ctx = []
    for k, j in chunk:
        i = pos[days[k]]
        ctx.append(torch.tensor(values[max(0, i - a.context) : i, j]))   # NaNs are allowed by Chronos
    with torch.no_grad():
        fc = pipe.predict(ctx, prediction_length=1, num_samples=a.num_samples)  # [B, S, 1]
    fc = fc[..., 0].float().numpy()
    for (k, j), s in zip(chunk, fc):
        out[k, j] = s
    if (b // a.batch) % 20 == 0:
        print(f"{b + len(chunk)}/{len(jobs)} forecasts, {time.time() - t0:.0f}s", flush=True)

os.makedirs("results/forecasts", exist_ok=True)
fn = f"results/forecasts/{a.name}_{a.table}_{a.period}.npz"
np.savez_compressed(fn, samples=out, dates=np.array([d.strftime("%Y-%m-%d") for d in days]),
                    industries=np.array(df.columns))
print("saved", fn, out.shape)

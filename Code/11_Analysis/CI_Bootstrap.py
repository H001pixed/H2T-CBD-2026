"""Non-inferiority CI for the 24 paired conditions, read from runs/result.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"


def _qwk(exp_id):
    p = RUNS / exp_id / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"missing run: {exp_id}")
    return json.loads(p.read_text(encoding="utf-8"))["test_qwk"]


# 24 paired conditions: delta-QWK = H2T-CBD - BlackBox.
deltas = []
for name in ("asap2", "dysf", "feat"):
    for s in (13, 42, 123):
        deltas.append(_qwk(f"B1_Pin_{name}_h2tcbd_s{s}") - _qwk(f"B1_Pin_{name}_blackbox_s{s}"))
for s in (13, 42, 123):
    deltas.append(_qwk(f"B1_LOPO_feat_h2tcbd_s{s}") - _qwk(f"B1_LOPO_feat_blackbox_s{s}"))
for src, tgt in (("asap2", "dysf"), ("asap2", "feat"), ("dysf", "asap2"), ("dysf", "feat")):
    for s in (13, 42, 123):
        deltas.append(_qwk(f"B1_cross_{src}2{tgt}_h2tcbd_s{s}") - _qwk(f"B1_cross_{src}2{tgt}_blackbox_s{s}"))

data = np.asarray(deltas, dtype=float)

n = len(data)
mean = np.mean(data)
std = np.std(data, ddof=1)
se = std / np.sqrt(n)
t_factor = stats.t.ppf(0.975, df=n - 1)
ci_normal_lower = mean - t_factor * se
ci_normal_upper = mean + t_factor * se

np.random.seed(42)
n_boot = 100000
boot_means = np.array([
    np.mean(np.random.choice(data, size=n, replace=True))
    for _ in range(n_boot)
])
ci_boot_lower = np.percentile(boot_means, 2.5)
ci_boot_upper = np.percentile(boot_means, 97.5)

print(f"n = {n}")
print(f"mean = {mean:.6f}")
print(f"std = {std:.4f}")
print(f"95% CI (t)  = [{ci_normal_lower:.6f}, {ci_normal_upper:.6f}]")
print(f"95% CI (boot) = [{ci_boot_lower:.6f}, {ci_boot_upper:.6f}]")
print(f"non-inferiority margin: -0.02")
print(f"lower bound {ci_normal_lower:.4f} vs margin -0.02: {'PASS' if ci_normal_lower > -0.02 else 'FAIL'}")

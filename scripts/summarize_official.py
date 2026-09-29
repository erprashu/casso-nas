"""Summarize finished official-code runs (runs/official) into one table.

For each seed and configuration it reports inherited-weight Kendall-tau, the
NAS-Bench-201 CIFAR-10 accuracy of the top-1 candidate by inherited-weight
accuracy (and the mean of the top 5), and the best accuracy in the seed's
candidate pool. If a re-scored file (rescored_<run>.json, written after
re-evaluating a saved .pth) exists, it is used instead of the run's own result.

    python scripts/summarize_official.py > runs/official/summary.txt
"""
import glob, json, os, re, statistics, time

OUT = "runs/official"
CONFIGS = ["gdas_vanilla", "gdas_casso", "uniform_vanilla", "uniform_casso"]
NAMES = {"gdas_vanilla": "GDAS", "gdas_casso": "GDAS + CASSO",
         "uniform_vanilla": "Uniform (SPOS)", "uniform_casso": "Uniform + CASSO"}


def load(seed, cfg):
    for path in (f"{OUT}/rescored_cifar10_s{seed}_{cfg}.json", f"{OUT}/cifar10_s{seed}_{cfg}_e250.json"):
        if os.path.exists(path):
            d = json.load(open(path))
            rows = sorted(d["rows"], key=lambda r: -r[2])
            tau = d.get("kendall_tau", d.get("kendall_tau_inherited"))
            return {"tau": tau, "top1": rows[0][1], "top5": sum(r[1] for r in rows[:5]) / 5,
                    "pool": max(r[1] for r in rows), "argmax": d.get("argmax_c10")}
    return None


seeds = sorted({int(m.group(1)) for f in glob.glob(f"{OUT}/*cifar10_s*_*.json")
                if (m := re.search(r"_s(\d+)_", f))})
res = {(s, c): load(s, c) for s in seeds for c in CONFIGS}
print(f"Official NAS-Bench-201 code, CIFAR-10, 250 epochs  (updated {time.strftime('%F %T')})\n")
for metric, label in (("tau", "Kendall-tau (inherited weights, 200 archs)"),
                      ("top1", "Top-1 pick by inherited accuracy: NB201 test acc (%)"),
                      ("top5", "Mean NB201 test acc of top-5 picks (%)")):
    print(label)
    print(f"  {'':18s}" + "".join(f"{'seed ' + str(s):>10s}" for s in seeds) + f"{'mean +- std':>18s}")
    for c in CONFIGS:
        vals = [res[s, c][metric] if res[s, c] else None for s in seeds]
        cells = "".join(f"{v:10.3f}" if metric == "tau" and v is not None else
                        f"{v:10.2f}" if v is not None else f"{'running':>10s}" for v in vals)
        done = [v for v in vals if v is not None]
        agg = (f"{statistics.mean(done):.3f} +- {statistics.stdev(done):.3f}" if metric == "tau" else
               f"{statistics.mean(done):.2f} +- {statistics.stdev(done):.2f}") if len(done) >= 2 else ""
        print(f"  {NAMES[c]:18s}{cells}{agg:>18s}")
    print()
pool = {s: next((res[s, c]["pool"] for c in CONFIGS if res[s, c]), None) for s in seeds}
print("Best architecture in each seed's 200-candidate pool: " +
      ", ".join(f"seed {s} {v:.2f}" for s, v in pool.items() if v is not None))

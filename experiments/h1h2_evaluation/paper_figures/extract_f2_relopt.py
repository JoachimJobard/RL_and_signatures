"""F2 data -- per-seed relative optimality eta for every (cell, learner, representation), read from the
clean-IC re-evaluation summaries. eta = cost_reduction / oracle_cost_reduction * 100. Raw values are
kept (divergences and nan included) so the figure clips the display and counts reliability honestly.
Writes one CSV consumed by the paper's F2 (box/violin) figure.

Usage: python .../extract_f2_relopt.py <out_csv>
"""
import os
import sys
import glob
import collections
import campaign as C

out_csv = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", ".") + "/f2_relopt_10seed.csv"
rows = ["cell,learner,representation,seed,oracle_cost_reduction,cost_reduction,eta_pct"]
n = 0
for name, cell in C.CELLS.items():
    orc = cell["oracle_cost_reduction"]
    for summ in sorted(glob.glob(f"{C.DATA_ROOT}/{cell['group_glob']}/reeval_clean/summary.tsv")):
        for ln in open(summ).read().strip().split("\n")[1:]:
            f = ln.split("\t")
            try:
                eta = f"{float(f[4]) / orc * 100:.4f}"
            except Exception:
                eta = "nan"
            rows.append(f"{name},{f[0]},{f[1]},{f[3]},{orc},{f[4]},{eta}"); n += 1
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
open(out_csv, "w").write("\n".join(rows) + "\n")
cnt = collections.Counter((r.split(",")[0], r.split(",")[1], r.split(",")[2]) for r in rows[1:])
bad = {k: v for k, v in cnt.items() if v != 10}
print(f"wrote {out_csv}  ({n} rows; cells {sorted(C.CELLS)}; combos != 10 seeds: {bad or 'none'})")

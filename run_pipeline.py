#!/usr/bin/env python
"""
End-to-end pipeline:  generate -> verify -> analyse -> report.

    python run_pipeline.py                 # verify committed data, run all analyses
    python run_pipeline.py --regenerate    # rebuild data/ from config first, then verify + analyse
    python run_pipeline.py --only q02 q04  # run selected analyses
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

STEPS = ["q00_data_profile", "q01_q1_churn_rate", "q02_definition_sensitivity", "q03_churn_drivers",
         "q04_support_and_churn", "q05_churn_prediction", "q06_golden_set"]


def verify(data: Path) -> bool:
    ok = True
    for line in (data / "CHECKSUMS.sha256").read_text().splitlines():
        h, f = line.split()
        got = hashlib.sha256((data / f).read_bytes()).hexdigest()
        print(f"  {'OK  ' if got == h else 'FAIL'} {f}")
        ok &= got == h
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="rebuild data/ from config/simulation.toml")
    ap.add_argument("--only", nargs="*", help="analysis ids to run, e.g. q01 q05")
    a = ap.parse_args()
    data = ROOT / "data"

    if a.regenerate:
        from churnsim import generate
        print("[1/3] generate data/ from config/simulation.toml")
        before = (data / "CHECKSUMS.sha256").read_text() if (data / "CHECKSUMS.sha256").exists() else None
        generate(out_dir=data)
        if before is not None and before != (data / "CHECKSUMS.sha256").read_text():
            print("  note: regenerated data differs from the previously committed version")
    print("[2/3] verify data/ checksums")
    if not verify(data):
        sys.exit("checksum mismatch: data/ was modified or produced by a different config/library version")

    steps = [s for s in STEPS if not a.only or s.split("_")[0] in a.only]
    print(f"[3/3] run {len(steps)} analyses")
    for s in steps:
        t = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "analysis" / f"{s}.py")], capture_output=True, text=True,
                           encoding="utf-8")
        if r.returncode:
            print(r.stdout, r.stderr)
            sys.exit(f"{s} failed")
        print(f"  {s:32s} {time.time() - t:5.1f}s")
    subprocess.run([sys.executable, str(ROOT / "analysis" / "build_report.py")], check=True)


if __name__ == "__main__":
    main()

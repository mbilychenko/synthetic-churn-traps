"""python -m churnsim [--config PATH] [--out DIR] [--seed N]"""
import argparse
from pathlib import Path

from . import generate

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic SaaS dataset.")
    ap.add_argument("--config", default=None, help="TOML config (default: config/simulation.toml)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=None, help="override the seed in the config")
    a = ap.parse_args()
    for f, h in generate(a.config, a.out, a.seed).items():
        print(f"{f:28s} sha256 {h[:16]}...")

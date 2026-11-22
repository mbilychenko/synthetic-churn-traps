"""Load the simulation config (TOML) and derive calendar helpers."""
from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "simulation.toml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class Calendar:
    """Months are indexed 0..n_months-1 from `first_month`; days are proleptic ordinals."""
    first_year: int
    first_month: int
    n_months: int
    export_ord: int

    @classmethod
    def from_config(cls, cfg: dict) -> "Calendar":
        y, m = map(int, cfg["calendar"]["first_month"].split("-"))
        exp = dt.date.fromisoformat(cfg["calendar"]["export_date"]).toordinal()
        return cls(y, m, int(cfg["calendar"]["n_months"]), exp)

    def month_start(self, idx: int) -> int:
        k = self.first_month - 1 + idx
        return dt.date(self.first_year + k // 12, k % 12 + 1, 1).toordinal()

    def month_of(self, day: int) -> int:
        d = dt.date.fromordinal(day)
        return (d.year - self.first_year) * 12 + d.month - self.first_month

    def month_index(self, yyyy_mm: str) -> int:
        y, m = map(int, yyyy_mm.split("-"))
        return (y - self.first_year) * 12 + m - self.first_month


def date_ord(iso: str) -> int:
    return dt.date.fromisoformat(iso).toordinal()

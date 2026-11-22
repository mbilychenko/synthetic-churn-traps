"""churnsim - simulate a fictional B2B SaaS company and export its data."""
from .config import load_config
from .export import TABLES, build_tables, write_tables
from .simulate import simulate

__all__ = ["load_config", "simulate", "build_tables", "write_tables", "TABLES", "generate"]
__version__ = "1.0.0"


def generate(config_path=None, out_dir="data", seed=None):
    """Simulate, export and return {file: sha256}."""
    cfg = load_config(config_path)
    tables, _ = build_tables(simulate(cfg, seed))
    return write_tables(tables, out_dir)

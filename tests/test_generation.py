"""The committed data must be exactly what the generator + config produce."""
import hashlib
from pathlib import Path

import pytest

from churnsim import generate

ROOT = Path(__file__).resolve().parents[1]


def committed_checksums():
    return dict(reversed(line.split()) for line in (ROOT / "data" / "CHECKSUMS.sha256").read_text().splitlines())


def test_committed_files_match_checksums():
    for f, h in committed_checksums().items():
        assert hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest() == h, f


@pytest.mark.slow
def test_regeneration_is_byte_identical(tmp_path):
    assert generate(out_dir=tmp_path) == committed_checksums()


@pytest.mark.slow
def test_seed_changes_the_data(tmp_path):
    assert generate(out_dir=tmp_path, seed=123) != committed_checksums()

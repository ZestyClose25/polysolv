"""Shared fixtures. Tests that need the raw Zhou file are skipped if it is missing."""

import pytest

from polysolv.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def zhou_long(cfg):
    from polysolv.data.load_zhou import load_zhou

    path = cfg["paths"]["zhou_xlsx"]
    if not path.exists():
        pytest.skip(f"raw Zhou file not found: {path}")
    z = cfg["zhou"]
    return load_zhou(path, z["sheet_main"], z["sheet_info"], z["rt_K"], z["polymer_rename"])

from pathlib import Path

import pytest

ARTIFACTS_ROOT = Path(__file__).parents[1] / "artifacts" / "benchmark_package"


@pytest.fixture
def pkg1_root() -> Path:
    return ARTIFACTS_ROOT / "001_customer_migration"


@pytest.fixture
def pkg2_root() -> Path:
    return ARTIFACTS_ROOT / "002_material_master_migration"


@pytest.fixture
def pkg3_root() -> Path:
    # No data/dataset.csv yet: this package has a noise_configuration.yaml
    # but no NoiseEngine has generated the noisy benchmark dataset yet, so
    # BenchmarkPackage.load() cannot be used for it (unlike pkg1/pkg2).
    return ARTIFACTS_ROOT / "003_supplier_migration"

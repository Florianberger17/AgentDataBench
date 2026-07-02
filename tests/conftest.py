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


@pytest.fixture
def pkg4_root() -> Path:
    # Same source_data.csv, synthesis_configuration.yaml and business_rules
    # as pkg3, but different noise_configuration.yaml probabilities. Used to
    # exercise PackageBuilder end-to-end.
    return ARTIFACTS_ROOT / "004_supplier_migration"


@pytest.fixture
def pkg5_root() -> Path:
    # No noise_configuration.yaml at all: exercises PackageBuilder's
    # no-noise-config-means-no-noise path. Its synthesis_configuration.yaml
    # only resynthesizes "part no." and "creator"; every other column uses
    # the "identity" strategy on purpose.
    return ARTIFACTS_ROOT / "005_material_master_migration"

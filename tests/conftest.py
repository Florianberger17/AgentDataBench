from pathlib import Path

import pytest

ARTIFACTS_ROOT = Path(__file__).parents[1] / "artifacts" / "benchmark_package"


# Package directories were renamed/consolidated to the
# <seq>_<domain>_<task_complexity>_<specification_completeness> scheme (see
# artifacts/benchmark_package/). Fixtures below are matched to their original
# role by task_id (stable across the rename), not by directory number.


@pytest.fixture
def pkg1_root() -> Path:
    return ARTIFACTS_ROOT / "001_customer_migration_basic_explicit"


@pytest.fixture
def pkg2_root() -> Path:
    return ARTIFACTS_ROOT / "009_material_master_migration_basic_explicit"


@pytest.fixture
def pkg3_root() -> Path:
    # Has a noise_configuration.yaml; task_id 003_ERP_SUPPLIER_MIGRATION.
    return ARTIFACTS_ROOT / "005_supplier_migration_basic_explicit"


@pytest.fixture
def pkg4_root() -> Path:
    # Same source_data.csv, synthesis_configuration.yaml and business_rules
    # as pkg3, but different noise_configuration.yaml probabilities. Used to
    # exercise PackageBuilder end-to-end. task_id 004_ERP_SUPPLIER_MIGRATION.
    return ARTIFACTS_ROOT / "099_supplier_migration"


@pytest.fixture
def pkg5_root() -> Path:
    # No noise_configuration.yaml at all: exercises PackageBuilder's
    # no-noise-config-means-no-noise path. Its synthesis_configuration.yaml
    # only resynthesizes "part no." and "creator"; every other column uses
    # the "identity" strategy on purpose. task_id 005_ERP_PART_MIGRATION.
    return ARTIFACTS_ROOT / "098_material_master_migration"

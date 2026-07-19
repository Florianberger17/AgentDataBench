import shutil

import yaml

from agentdatabench.domain.benchmark_package import BenchmarkPackage


def test_load_pkg1_customer_migration(pkg1_root):
    pkg = BenchmarkPackage.load(pkg1_root)

    assert pkg.scenario.scenario_id == "001_ERP_CUSTOMER_MIGRATION"
    assert pkg.task.task_id == "001_ERP_CUSTOMER_MIGRATION"
    assert pkg.metadata.version == "1.0"
    assert pkg.source_schema.table == "customer_master"
    assert pkg.target_schema.table == "customer_master"
    assert not pkg.dataset.df.empty
    assert not pkg.clean_dataset.df.empty
    assert not pkg.ground_truth.df.empty


def _underspecified_variant(pkg_root, tmp_path):
    """Copies a real (explicit) package and turns it into an underspecified
    one: drops the schema references from task.yaml in favor of a small
    target_example.csv built from the first two ground_truth.csv rows -
    mirrors how a hand-authored level 2/4 package would look."""
    work_root = tmp_path / "pkg"
    shutil.copytree(pkg_root, work_root)

    ground_truth_lines = (work_root / "ground_truth" / "ground_truth.csv").read_text().splitlines()
    target_example_path = work_root / "data" / "target_example.csv"
    target_example_path.write_text("\n".join(ground_truth_lines[:3]) + "\n")

    task_path = work_root / "task.yaml"
    task_data = yaml.safe_load(task_path.read_text())
    del task_data["input"]["source_schema"]
    del task_data["input"]["target_schema"]
    task_data["input"]["target_example"] = "data/target_example.csv"
    task_path.write_text(yaml.safe_dump(task_data))

    meta_path = work_root / "metadata.yaml"
    meta_data = yaml.safe_load(meta_path.read_text())
    meta_data["specification_completeness"] = "underspecified"
    meta_path.write_text(yaml.safe_dump(meta_data))

    return work_root


def test_load_underspecified_package_has_no_schemas_but_has_target_example(pkg1_root, tmp_path):
    work_root = _underspecified_variant(pkg1_root, tmp_path)

    pkg = BenchmarkPackage.load(work_root)

    assert pkg.source_schema is None
    assert pkg.target_schema is None
    assert pkg.target_example is not None
    assert len(pkg.target_example.df) == 2


def test_load_pkg2_material_master_migration(pkg2_root):
    pkg = BenchmarkPackage.load(pkg2_root)

    assert pkg.scenario.scenario_id == "002_ERP_PART_MIGRATION"
    assert pkg.task.task_id == "002_ERP_PART_MIGRATION"

    mappings = pkg.task.business_rules.mappings
    assert mappings is not None
    by_target = {m.target_field: m for m in mappings}
    assert by_target["erp no"].transformation.type == "sequential_number"
    assert by_target["erp no"].transformation.start == 30000000000
    assert by_target["description"].transformation.type == "concatenate"
    assert by_target["material type"].transformation.type == "value_mapping"

    assert not pkg.dataset.df.empty
    assert not pkg.clean_dataset.df.empty
    assert not pkg.ground_truth.df.empty

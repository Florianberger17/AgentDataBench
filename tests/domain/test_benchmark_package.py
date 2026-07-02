from migratebench.domain.benchmark_package import BenchmarkPackage


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

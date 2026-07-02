from migratebench.domain.noise_configuration import NoiseConfiguration

VALID = {
    "seed": 42,
    "excluded_columns": ["supplier no", "name 1"],
    "noise_types": [
        {"type": "typo", "probability": 0.05},
        {"type": "duplicate", "probability": 0.10},
        {"type": "missing_value", "probability": 0.03},
    ],
}


def test_noise_configuration_happy_path():
    config = NoiseConfiguration(**VALID)
    assert config.seed == 42
    assert config.excluded_columns == ["supplier no", "name 1"]
    assert [nt.type for nt in config.noise_types] == ["typo", "duplicate", "missing_value"]
    assert config.noise_types[0].probability == 0.05


def test_noise_configuration_defaults_excluded_columns_to_empty():
    data = {k: v for k, v in VALID.items() if k != "excluded_columns"}
    config = NoiseConfiguration(**data)
    assert config.excluded_columns == []


def test_noise_type_config_allows_extra_fields():
    config = NoiseConfiguration(
        seed=1,
        noise_types=[{"type": "custom_noise", "probability": 0.2, "intensity": "high"}],
    )
    assert config.noise_types[0].intensity == "high"

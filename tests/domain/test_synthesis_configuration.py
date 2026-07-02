from migratebench.domain.synthesis_configuration import SynthesisConfiguration

VALID = {
    "seed": 123,
    "locale": "de_DE",
    "columns": [
        {"column": "supplier no", "strategy": "unique_sequence", "format": "{:07d}", "start": 1000000},
        {"column": "name 1", "strategy": "faker", "provider": "company"},
    ],
}


def test_synthesis_configuration_happy_path():
    config = SynthesisConfiguration(**VALID)
    assert config.seed == 123
    assert config.locale == "de_DE"
    assert config.columns[0].column == "supplier no"
    assert config.columns[0].format == "{:07d}"
    assert config.columns[1].provider == "company"


def test_synthesis_configuration_defaults_locale():
    data = {k: v for k, v in VALID.items() if k != "locale"}
    config = SynthesisConfiguration(**data)
    assert config.locale == "de_DE"

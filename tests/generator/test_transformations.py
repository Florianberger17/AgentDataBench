import pandas as pd
import pytest

from agentdatabench.domain.task import MappingRule
from agentdatabench.generator.transformations import (
    CalculationHandler,
    ConcatenateHandler,
    ConditionalValueHandler,
    ConstantHandler,
    CopyHandler,
    DateDifferenceHandler,
    DateFormatHandler,
    LookupHandler,
    MissingTargetColumn,
    NumericFactorHandler,
    BoundedValueHandler,
    NumericOffsetHandler,
    RangeMappingHandler,
    RoundHandler,
    SerialDateConversionHandler,
    TruncateHandler,
    UnitConversionHandler,
    PrefixAndPadHandler,
    SequenceHandler,
    SequentialNumberHandler,
    TransformationContext,
    UppercaseHandler,
    ValueMappingHandler,
    _to_numeric,
)


SUPPLIER_MAPPING = pd.DataFrame(
    {
        "Supplier No. Legacy System": ["2787004", "2787012"],
        "Supplier Number New": ["513001", "513002"],
    }
)


def _lookup_mapping(**overrides):
    transformation = {
        "type": "lookup",
        "reference": "supplier_mapping",
        "lookup_key": "Supplier No. Legacy System",
        "return_field": "Supplier Number New",
    }
    transformation.update(overrides)
    return MappingRule(
        source_field="SupplierNo", target_field="VENDOR", transformation=transformation
    )


def _context(reference=None):
    return TransformationContext(
        reference_data={"supplier_mapping": SUPPLIER_MAPPING if reference is None else reference}
    )


def test_copy_handler():
    df = pd.DataFrame({"src": ["a", "b"]})
    mapping = MappingRule(
        source_field="src", target_field="dst", transformation={"type": "copy"}
    )
    result = CopyHandler().apply(df, mapping)
    assert list(result) == ["a", "b"]


def test_concatenate_handler_preserves_internal_spacing():
    df = pd.DataFrame({"a": ["1000-0001"], "b": ["slide  complete"]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="description",
        transformation={"type": "concatenate", "separator": " "},
    )
    result = ConcatenateHandler().apply(df, mapping)
    assert list(result) == ["1000-0001 slide  complete"]


def test_value_mapping_handler_maps_known_values():
    df = pd.DataFrame({"country_code": ["DE", "AT", "CH"]})
    mapping = MappingRule(
        source_field="country_code",
        target_field="country",
        transformation={
            "type": "value_mapping",
            "mapping": {"DE": "DEU", "AT": "AUT", "CH": "CHE"},
        },
    )
    result = ValueMappingHandler().apply(df, mapping)
    assert list(result) == ["DEU", "AUT", "CHE"]


def test_value_mapping_handler_raises_on_unmapped_value():
    df = pd.DataFrame({"country_code": ["DE", "FR"]})
    mapping = MappingRule(
        source_field="country_code",
        target_field="country",
        transformation={"type": "value_mapping", "mapping": {"DE": "DEU"}},
    )
    with pytest.raises(ValueError, match="FR"):
        ValueMappingHandler().apply(df, mapping)


def test_date_format_handler_iso_to_ddmmyyyy():
    df = pd.DataFrame({"date": ["2025-01-30", "2026-03-19"]})
    mapping = MappingRule(
        source_field="date",
        target_field="out_date",
        transformation={
            "type": "date_format",
            "input_format": "YYYY-MM-DD",
            "output_format": "DDMMYYYY",
        },
    )
    result = DateFormatHandler().apply(df, mapping)
    assert list(result) == ["30012025", "19032026"]


def test_date_format_handler_dotted_two_digit_year():
    df = pd.DataFrame({"date": ["19.04.01"]})
    mapping = MappingRule(
        source_field="date",
        target_field="out_date",
        transformation={
            "type": "date_format",
            "input_format": "DD.MM.YY",
            "output_format": "DD.MM.YYYY",
        },
    )
    result = DateFormatHandler().apply(df, mapping)
    assert list(result) == ["19.04.2001"]


def test_sequential_number_handler_start_increment_digits():
    df = pd.DataFrame({"x": [0, 0, 0]})
    mapping = MappingRule(
        source_field="x",
        target_field="erp_no",
        transformation={
            "type": "sequential_number",
            "start": 30000000000,
            "increment": 1,
            "digits": 11,
        },
    )
    result = SequentialNumberHandler().apply(df, mapping)
    assert list(result) == ["30000000000", "30000000001", "30000000002"]
    assert len(set(result)) == len(result)


def test_concatenate_handler_renders_pattern_with_mixed_separators():
    df = pd.DataFrame(
        {
            "OrderType": ["SPT"],
            "CustomerTransaction": ["214000"],
            "CustomerName": ["Döring Tschentscher KG"],
        }
    )
    mapping = MappingRule(
        source_fields=["OrderType", "CustomerTransaction", "CustomerName"],
        target_field="Short Description",
        transformation={
            "type": "concatenate",
            "pattern": "{OrderType} {CustomerTransaction}-{CustomerName}",
        },
    )
    result = ConcatenateHandler().apply(df, mapping)
    assert list(result) == ["SPT 214000-Döring Tschentscher KG"]


def test_concatenate_handler_pattern_supports_field_names_with_spaces():
    df = pd.DataFrame({"Order Currency": ["EUR"], "Plant": ["100"]})
    mapping = MappingRule(
        source_fields=["Order Currency", "Plant"],
        target_field="label",
        transformation={"type": "concatenate", "pattern": "{Order Currency}/{Plant}"},
    )
    result = ConcatenateHandler().apply(df, mapping)
    assert list(result) == ["EUR/100"]


def test_sequence_handler_prefixes_running_number():
    df = pd.DataFrame({"x": [0, 0, 0]})
    mapping = MappingRule(
        target_field="Project Definition",
        transformation={"type": "sequence", "prefix": "P-", "start_value": 310000, "increment": 1},
    )
    result = SequenceHandler().apply(df, mapping)
    assert list(result) == ["P-310000", "P-310001", "P-310002"]
    assert len(set(result)) == len(result)


def test_sequence_handler_accepts_start_as_alias_and_pads_digits():
    df = pd.DataFrame({"x": [0, 0]})
    mapping = MappingRule(
        target_field="id",
        transformation={"type": "sequence", "prefix": "P-", "start": 7, "digits": 4},
    )
    result = SequenceHandler().apply(df, mapping)
    assert list(result) == ["P-0007", "P-0008"]


def test_constant_handler_fills_every_row():
    df = pd.DataFrame({"x": [0, 0, 0]})
    mapping = MappingRule(
        target_field="Company Code", transformation={"type": "constant", "value": "0898"}
    )
    result = ConstantHandler().apply(df, mapping)
    assert list(result) == ["0898", "0898", "0898"]


def test_constant_handler_stringifies_unquoted_yaml_integer():
    """`value: 1` must not turn the column numeric - the CSV round-trip reads
    every column back as a string."""
    df = pd.DataFrame({"x": [0, 0]})
    mapping = MappingRule(
        target_field="Factory Calendar Key", transformation={"type": "constant", "value": 1}
    )
    result = ConstantHandler().apply(df, mapping)
    assert list(result) == ["1", "1"]


def test_prefix_and_pad_handler_pads_then_prefixes():
    df = pd.DataFrame({"TransactionNo": ["103", "1037", "21947"]})
    mapping = MappingRule(
        source_field="TransactionNo",
        target_field="PO_NUMBER",
        transformation={
            "type": "prefix_and_pad",
            "pad_to": 5,
            "pad_character": "0",
            "prefix": "97000",
        },
    )
    result = PrefixAndPadHandler().apply(df, mapping)
    assert list(result) == ["9700000103", "9700001037", "9700021947"]


def test_uppercase_handler_normalises_mixed_case():
    df = pd.DataFrame({"Incoterm": ["fca", "FCA", "DAP"]})
    mapping = MappingRule(
        source_field="Incoterm", target_field="INCOTERMS1", transformation={"type": "uppercase"}
    )
    result = UppercaseHandler().apply(df, mapping)
    assert list(result) == ["FCA", "FCA", "DAP"]


def test_copy_handler_substitutes_default_for_empty_values():
    df = pd.DataFrame({"IncotermLocation": ["Grafenau", None, "  "]})
    mapping = MappingRule(
        source_field="IncotermLocation",
        target_field="INCOTERMS2",
        transformation={"type": "copy", "default_if_empty": "0"},
    )
    result = CopyHandler().apply(df, mapping)
    assert list(result) == ["Grafenau", "0", "0"]


def test_lookup_handler_resolves_and_pads():
    df = pd.DataFrame({"SupplierNo": ["2787004", "2787012"]})
    mapping = _lookup_mapping(
        post_processing={"type": "pad", "pad_to": 10, "pad_character": "0"}
    )
    result = LookupHandler().apply(df, mapping, _context())
    assert list(result) == ["0000513001", "0000513002"]


def test_lookup_handler_raises_on_ambiguous_reference_table():
    """The same legacy key carrying two different target values would make the
    ground truth depend on the reference file's row order."""
    ambiguous = pd.DataFrame(
        {
            "Supplier No. Legacy System": ["2787004", "2787004"],
            "Supplier Number New": ["513001", "513099"],
        }
    )
    df = pd.DataFrame({"SupplierNo": ["2787004"]})
    with pytest.raises(ValueError, match="ambiguous"):
        LookupHandler().apply(df, _lookup_mapping(), _context(ambiguous))


def test_lookup_handler_raises_when_reference_data_missing():
    df = pd.DataFrame({"SupplierNo": ["2787004"]})
    with pytest.raises(ValueError, match="not available"):
        LookupHandler().apply(df, _lookup_mapping(), TransformationContext())


def test_lookup_handler_reports_unresolvable_rows_for_exclusion():
    df = pd.DataFrame({"SupplierNo": ["2787004", "9999999"]})
    mapping = _lookup_mapping(on_missing="exclude_record")
    assert list(LookupHandler().excluded_rows(df, mapping, _context())) == [False, True]


def test_lookup_handler_excludes_nothing_without_on_missing():
    df = pd.DataFrame({"SupplierNo": ["2787004", "9999999"]})
    assert not LookupHandler().excluded_rows(df, _lookup_mapping(), _context()).any()


def test_numeric_offset_handler_keeps_large_numbers_readable():
    """A document number must not come out in scientific notation."""
    df = pd.DataFrame({"TransactionNo": ["21910", "31149"]})
    mapping = MappingRule(
        source_field="TransactionNo",
        target_field="PO_NUMBER",
        transformation={"type": "numeric_offset", "offset": 9700000000},
    )
    result = NumericOffsetHandler().apply(df, mapping)
    assert list(result) == ["9700021910", "9700031149"]


def test_numeric_factor_handler_renumbers_in_steps():
    df = pd.DataFrame({"ItemNo": ["1", "2", "28"]})
    mapping = MappingRule(
        source_field="ItemNo",
        target_field="PO_ITEM",
        transformation={"type": "numeric_factor", "factor": 10},
    )
    result = NumericFactorHandler().apply(df, mapping)
    assert list(result) == ["10", "20", "280"]


def test_calculation_handler_preserves_the_decimal_comma():
    """German-locale source columns must not silently become dot-separated."""
    df = pd.DataFrame({"OrderPrice": ["41274,41", "2,50"], "OpenQuantity": ["595,69", "4"]})
    mapping = MappingRule(
        source_fields=["OrderPrice", "OpenQuantity"],
        target_field="GR_PRICE",
        transformation={
            "type": "calculation",
            "formula": "OrderPrice * OpenQuantity",
            "decimals": 2,
        },
    )
    result = CalculationHandler().apply(df, mapping)
    assert list(result) == ["24586753,29", "10,00"]


def test_calculation_handler_keeps_dot_separated_input_dotted():
    df = pd.DataFrame({"a": ["2.5"], "b": ["4"]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="c",
        transformation={"type": "calculation", "formula": "a * b", "decimals": 2},
    )
    assert list(CalculationHandler().apply(df, mapping)) == ["10.00"]


def test_conditional_value_handler_picks_first_matching_condition():
    df = pd.DataFrame({"CostCenter": ["0417-2005", None, "  "]})
    mapping = MappingRule(
        source_field="CostCenter",
        target_field="ACCTASSCAT",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {"when": {"field": "CostCenter", "is": "not_empty"}, "value": "K"},
                {"value": "P"},
            ],
        },
    )
    result = ConditionalValueHandler().apply(df, mapping)
    assert list(result) == ["K", "P", "P"]


def test_conditional_value_handler_takes_a_value_from_another_column():
    """`from_field` is the distinction prose like `value: "Description1"`
    could not express: the column's value, not the literal string."""
    df = pd.DataFrame(
        {"MaterialNo": ["3001-6944", None], "Description1": ["shaft", "consulting"]}
    )
    mapping = MappingRule(
        source_fields=["Description1", "MaterialNo"],
        target_field="SHORT_TEXT",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {"when": {"field": "MaterialNo", "is": "empty"}, "from_field": "Description1"},
                {"value": ""},
            ],
        },
    )
    result = ConditionalValueHandler().apply(df, mapping)
    assert list(result) == ["", "consulting"]


def test_conditional_value_handler_supports_equals_test():
    df = pd.DataFrame({"Status": ["open", "closed"]})
    mapping = MappingRule(
        source_field="Status",
        target_field="flag",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {"when": {"field": "Status", "is": "equals", "to": "open"}, "value": "X"},
                {"value": "-"},
            ],
        },
    )
    assert list(ConditionalValueHandler().apply(df, mapping)) == ["X", "-"]


def test_conditional_value_handler_defers_on_unknown_field():
    """A condition may test a *target* field, which does not exist yet while
    the earlier fields are being built. The handler cannot tell that from a
    typo, so it defers either way and GroundTruthCreator reports a name
    nobody ever builds as an unresolvable dependency."""
    df = pd.DataFrame({"a": ["1"]})
    mapping = MappingRule(
        source_field="a",
        target_field="x",
        transformation={
            "type": "conditional_value",
            "conditions": [{"when": {"field": "nope", "is": "empty"}, "value": "y"}],
        },
    )
    with pytest.raises(MissingTargetColumn, match="nope"):
        ConditionalValueHandler().apply(df, mapping)


def _range_mapping(ranges, source_field="revenue"):
    return MappingRule(
        source_field=source_field,
        target_field="limit",
        transformation={"type": "range_mapping", "ranges": ranges},
    )


_LIMIT_RANGES = [
    {"to": 5000, "value": "1"},
    {"from": 5000, "to": 10000, "value": "3.000"},
    {"from": 10000, "value": "10.000"},
]


def test_range_mapping_picks_the_bracket_the_value_falls_into():
    df = pd.DataFrame({"revenue": ["234,00", "6.441,89", "216.424,12"]})
    result = RangeMappingHandler().apply(df, _range_mapping(_LIMIT_RANGES))
    assert list(result) == ["1", "3.000", "10.000"]


def test_range_mapping_bounds_are_lower_inclusive_and_upper_exclusive():
    """Adjacent brackets share a boundary; the value on it belongs to the
    upper one."""
    df = pd.DataFrame({"revenue": ["4.999,99", "5.000,00", "9.999,99", "10.000,00"]})
    result = RangeMappingHandler().apply(df, _range_mapping(_LIMIT_RANGES))
    assert list(result) == ["1", "3.000", "3.000", "10.000"]


def test_range_mapping_reads_a_target_field_too():
    df = pd.DataFrame({"other": ["x"]})
    context = TransformationContext(target_columns={"revenue": pd.Series(["7.000,00"])})
    result = RangeMappingHandler().apply(df, _range_mapping(_LIMIT_RANGES), context)
    assert list(result) == ["3.000"]


def test_range_mapping_defers_until_the_target_field_it_reads_exists():
    df = pd.DataFrame({"other": ["x"]})
    with pytest.raises(MissingTargetColumn):
        RangeMappingHandler().apply(df, _range_mapping(_LIMIT_RANGES))


def test_range_mapping_raises_for_a_value_no_bracket_covers():
    """Brackets closed on both sides leave a gap, and a value falling into it
    would otherwise come out silently empty."""
    df = pd.DataFrame({"revenue": ["20,00"]})
    closed = [{"from": 0, "to": 10, "value": "low"}]
    with pytest.raises(ValueError, match="covers no bracket"):
        RangeMappingHandler().apply(df, _range_mapping(closed))


def test_range_mapping_raises_for_a_bracket_without_bounds():
    df = pd.DataFrame({"revenue": ["1,00"]})
    with pytest.raises(ValueError, match="bounds nothing"):
        RangeMappingHandler().apply(df, _range_mapping([{"value": "1"}]))


def test_range_mapping_leaves_a_non_numeric_value_empty():
    df = pd.DataFrame({"revenue": [""]})
    result = RangeMappingHandler().apply(df, _range_mapping(_LIMIT_RANGES))
    assert list(result) == [""]



def _bounded(**transformation):
    return MappingRule(
        source_field="costs",
        target_field="WIP",
        transformation={"type": "bounded_value", "decimals": 2, **transformation},
    )


def test_bounded_value_caps_at_a_bound_read_from_another_column():
    df = pd.DataFrame({"costs": ["5000,00", "200,00"], "cap": ["1000,00", "900,00"]})
    result = BoundedValueHandler().apply(df, _bounded(maximum_field="cap"))
    assert list(result) == ["1000,00", "200,00"]


def test_bounded_value_scales_a_bound_carried_with_the_opposite_sign():
    """The planned revenue is posted as a credit, so the limit it imposes is
    its negation."""
    df = pd.DataFrame({"costs": ["5000,00"], "revenue": ["-1000,00"]})
    mapping = _bounded(maximum_field="revenue", maximum_factor=-1)
    assert list(BoundedValueHandler().apply(df, mapping)) == ["1000,00"]


def test_bounded_value_applies_the_minimum_last_so_the_floor_wins():
    """A project with no planned revenue has a cap of zero; its work in
    progress is zero, not a negative figure."""
    df = pd.DataFrame({"costs": ["142,00"], "cap": ["0,00"]})
    mapping = _bounded(maximum_field="cap", minimum=0)
    assert list(BoundedValueHandler().apply(df, mapping)) == ["0,00"]


def test_bounded_value_lifts_a_value_to_a_literal_minimum():
    df = pd.DataFrame({"costs": ["-30,00", "40,00"]})
    assert list(BoundedValueHandler().apply(df, _bounded(minimum=0))) == [
        "0,00",
        "40,00",
    ]


def test_bounded_value_reads_value_and_bound_from_target_fields():
    df = pd.DataFrame({"other": ["x"]})
    context = TransformationContext(
        target_columns={
            "costs": pd.Series(["900,00"]),
            "cap": pd.Series(["500,00"]),
        }
    )
    mapping = _bounded(maximum_field="cap")
    assert list(BoundedValueHandler().apply(df, mapping, context)) == ["500,00"]


def test_bounded_value_defers_until_the_bound_it_reads_exists():
    df = pd.DataFrame({"costs": ["900,00"]})
    with pytest.raises(MissingTargetColumn):
        BoundedValueHandler().apply(df, _bounded(maximum_field="cap"))


def test_bounded_value_without_any_bound_raises():
    df = pd.DataFrame({"costs": ["1,00"]})
    with pytest.raises(ValueError, match="nothing to bound"):
        BoundedValueHandler().apply(df, _bounded())


def test_bounded_value_with_both_a_literal_and_a_field_bound_raises():
    df = pd.DataFrame({"costs": ["1,00"], "cap": ["2,00"]})
    mapping = _bounded(maximum=10, maximum_field="cap")
    with pytest.raises(ValueError, match="reads one of them"):
        BoundedValueHandler().apply(df, mapping)


def _behaviour_clause(**overrides):
    clause = {
        "field": "customer no.",
        "reference": "cross_reference",
        "key_field": "legacy",
        "return_field": "behaviour",
        "is": "equals",
        "to": "good payer",
    }
    clause.update(overrides)
    return clause


def _cross_reference_context(target_columns=None):
    return TransformationContext(
        reference_data={
            "cross_reference": pd.DataFrame(
                {
                    "legacy": ["100", "200", "300"],
                    "behaviour": ["good payer", "poor payer", "poor payer/blocked"],
                }
            )
        },
        target_columns=target_columns or {},
    )


def test_conditional_value_resolves_the_tested_field_through_a_reference_table():
    """Branching on an attribute the dataset does not carry at all - the
    payment behaviour lives only in the cross reference list."""
    df = pd.DataFrame({"customer no.": ["100", "200", "300", "999"]})
    mapping = MappingRule(
        target_field="class",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {"when": _behaviour_clause(), "value": "A"},
                {
                    "when": _behaviour_clause(to="poor payer/blocked"),
                    "value": "C",
                },
                {"value": "B"},
            ],
        },
    )
    result = ConditionalValueHandler().apply(df, mapping, _cross_reference_context())
    # "999" is absent from the table, so it resolves to no behaviour at all
    # and falls through to the fallback.
    assert list(result) == ["A", "B", "C", "B"]


def test_conditional_value_all_of_requires_every_clause_to_hold():
    df = pd.DataFrame({"customer no.": ["100", "100", "200"]})
    context = _cross_reference_context(
        target_columns={"limit": pd.Series(["50.000", "1", "50.000"])}
    )
    mapping = MappingRule(
        target_field="class",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {
                    "all_of": [
                        _behaviour_clause(),
                        {"field": "limit", "is": "in", "to": ["50.000"]},
                    ],
                    "value": "AA",
                },
                {"when": _behaviour_clause(), "value": "A"},
                {"value": "C"},
            ],
        },
    )
    assert list(ConditionalValueHandler().apply(df, mapping, context)) == [
        "AA",
        "A",
        "C",
    ]


def test_conditional_value_takes_its_value_from_a_target_field():
    df = pd.DataFrame({"a": ["1", "2"]})
    context = TransformationContext(target_columns={"b": pd.Series(["x", "y"])})
    mapping = MappingRule(
        target_field="c",
        transformation={
            "type": "conditional_value",
            "conditions": [{"when": {"field": "a", "is": "not_empty"}, "from_field": "b"}],
        },
    )
    assert list(ConditionalValueHandler().apply(df, mapping, context)) == ["x", "y"]


def test_lookup_handler_leaves_empty_source_values_empty():
    """Nothing to resolve is not a lookup failure - the target attribute is
    optional in that case."""
    df = pd.DataFrame({"SupplierNo": ["2787004", None, "  "]})
    mapping = _lookup_mapping(
        post_processing={"type": "pad", "pad_to": 10, "pad_character": "0"}
    )
    result = LookupHandler().apply(df, mapping, _context())
    assert list(result) == ["0000513001", "", ""]


def test_lookup_handler_keep_empty_tolerates_unknown_keys():
    df = pd.DataFrame({"SupplierNo": ["2787004", "9999999"]})
    mapping = _lookup_mapping(on_missing="keep_empty")
    assert list(LookupHandler().apply(df, mapping, _context())) == ["513001", ""]


def test_lookup_handler_does_not_exclude_rows_with_an_empty_key():
    df = pd.DataFrame({"SupplierNo": ["2787004", None, "9999999"]})
    mapping = _lookup_mapping(on_missing="exclude_record")
    assert list(LookupHandler().excluded_rows(df, mapping, _context())) == [False, False, True]


def test_value_mapping_handler_default_copy_adopts_unmapped_values():
    """A table that only renames the units it has to rename - everything else
    passes through instead of being rejected."""
    df = pd.DataFrame({"bom unit": ["PCS", "MM", "M"]})
    mapping = MappingRule(
        source_field="bom unit",
        target_field="MEINS_POS",
        transformation={
            "type": "value_mapping",
            "mapping": {"PCS": "ST", "GR": "G"},
            "default": "copy",
        },
    )
    assert list(ValueMappingHandler().apply(df, mapping)) == ["ST", "MM", "M"]


def test_value_mapping_handler_default_literal_replaces_unmapped_values():
    df = pd.DataFrame({"u": ["PCS", "XX"]})
    mapping = MappingRule(
        source_field="u",
        target_field="v",
        transformation={"type": "value_mapping", "mapping": {"PCS": "ST"}, "default": "?"},
    )
    assert list(ValueMappingHandler().apply(df, mapping)) == ["ST", "?"]


def test_conditional_value_handler_tests_membership_in_a_value_list():
    df = pd.DataFrame(
        {"bom unit": ["MM", "PCS", "M"], "length": ["6238,16", "0", "12,5"], "qty": ["1", "3", "9"]}
    )
    mapping = MappingRule(
        source_fields=["bom unit", "length", "qty"],
        target_field="MENGE",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {"when": {"field": "bom unit", "is": "in", "to": ["MM", "M"]}, "from_field": "length"},
                {"from_field": "qty"},
            ],
        },
    )
    result = ConditionalValueHandler().apply(df, mapping)
    assert list(result) == ["6238,16", "3", "12,5"]


def test_conditional_value_handler_tests_membership_in_a_reference_table():
    df = pd.DataFrame({"component no.": ["2787004", "9999999"]})
    reference = pd.DataFrame({"part no.": ["2787004"], "material no.": ["1"]})
    mapping = MappingRule(
        source_field="component no.",
        target_field="POSTP",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {
                    "when": {
                        "field": "component no.",
                        "is": "in_reference",
                        "reference": "materials",
                        "key_field": "part no.",
                    },
                    "value": "L",
                },
                {"value": "T"},
            ],
        },
    )
    context = TransformationContext(reference_data={"materials": reference})
    assert list(ConditionalValueHandler().apply(df, mapping, context)) == ["L", "T"]


def test_conditional_value_handler_not_in_reference_is_the_inverse():
    df = pd.DataFrame({"c": ["a", "b"], "text": ["keep", "keep"]})
    reference = pd.DataFrame({"k": ["a"]})
    mapping = MappingRule(
        source_fields=["c", "text"],
        target_field="POTX1",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {
                    "when": {
                        "field": "c",
                        "is": "not_in_reference",
                        "reference": "r",
                        "key_field": "k",
                    },
                    "from_field": "text",
                },
                {"value": ""},
            ],
        },
    )
    context = TransformationContext(reference_data={"r": reference})
    assert list(ConditionalValueHandler().apply(df, mapping, context)) == ["", "keep"]


def test_conditional_value_handler_raises_on_unknown_reference_table():
    df = pd.DataFrame({"c": ["a"]})
    mapping = MappingRule(
        source_field="c",
        target_field="x",
        transformation={
            "type": "conditional_value",
            "conditions": [
                {
                    "when": {"field": "c", "is": "in_reference", "reference": "nope", "key_field": "k"},
                    "value": "y",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="not available"):
        ConditionalValueHandler().apply(df, mapping, TransformationContext())


def test_truncate_handler_cuts_to_the_target_field_length():
    df = pd.DataFrame({"description": ["x" * 50, "short"]})
    mapping = MappingRule(
        source_field="description",
        target_field="MATERIAL_DESC",
        transformation={"type": "truncate", "length": 40},
    )
    result = TruncateHandler().apply(df, mapping)
    assert [len(value) for value in result] == [40, 5]


def test_truncate_handler_cuts_an_already_built_target_field():
    df = pd.DataFrame({"legacy name": ["ignored"]})
    mapping = MappingRule(
        source_field="DESCRIPT",
        target_field="NAME",
        transformation={"type": "truncate", "length": 20},
    )
    context = TransformationContext(
        target_columns={"DESCRIPT": pd.Series(["Mechanical Engineering"])}
    )
    result = TruncateHandler().apply(df, mapping, context)
    assert list(result) == ["Mechanical Engineeri"]


def test_truncate_handler_defers_on_an_unknown_field():
    df = pd.DataFrame({"description": ["anything"]})
    mapping = MappingRule(
        source_field="DESCRIPT",
        target_field="NAME",
        transformation={"type": "truncate", "length": 20},
    )
    with pytest.raises(MissingTargetColumn):
        TruncateHandler().apply(df, mapping)


def test_round_handler_keeps_the_decimal_comma():
    df = pd.DataFrame({"price": ["11629,426", "5,281"]})
    mapping = MappingRule(
        source_field="price",
        target_field="COND_AMOUNT",
        transformation={"type": "round", "decimals": 2},
    )
    assert list(RoundHandler().apply(df, mapping)) == ["11629,43", "5,28"]


def test_serial_date_conversion_handler_converts_an_excel_serial():
    df = pd.DataFrame({"valid from date": ["37681"]})
    mapping = MappingRule(
        source_field="valid from date",
        target_field="VALID_FROM",
        transformation={
            "type": "serial_date_conversion",
            "epoch": "1899-12-30",
            "output_format": "DD.MM.YYYY",
        },
    )
    assert list(SerialDateConversionHandler().apply(df, mapping)) == ["01.03.2003"]


def test_serial_date_conversion_handler_falls_back_for_empty_serials():
    df = pd.DataFrame({"s": ["37681", None, "  "]})
    mapping = MappingRule(
        source_field="s",
        target_field="VALID_FROM",
        transformation={
            "type": "serial_date_conversion",
            "epoch": "1899-12-30",
            "output_format": "DD.MM.YYYY",
            "default_if_empty": "01.06.2025",
        },
    )
    result = SerialDateConversionHandler().apply(df, mapping)
    assert list(result) == ["01.03.2003", "01.06.2025", "01.06.2025"]


def test_value_mapping_handler_matches_unquoted_yaml_integer_keys():
    """A dataset column is always text, a bare YAML key is an int - without
    normalising, `5303: "01"` would never match."""
    df = pd.DataFrame({"price list no.": ["5303", "9503"]})
    mapping = MappingRule(
        source_field="price list no.",
        target_field="PRICE_LIST",
        transformation={"type": "value_mapping", "mapping": {5303: "01", 9503: "02"}},
    )
    assert list(ValueMappingHandler().apply(df, mapping)) == ["01", "02"]


def test_sequential_number_handler_assigns_one_number_per_group():
    """All items of one order share a document number."""
    df = pd.DataFrame({"order": ["A", "A", "B", "B", "B"], "item": list("12345")})
    mapping = MappingRule(
        target_field="VBELN",
        transformation={"type": "sequential_number", "assigned_per": "group", "start": 1000},
    )
    context = TransformationContext(group_keys=["order"])

    result = SequentialNumberHandler().apply(df, mapping, context)

    assert list(result) == ["1000", "1000", "1001", "1001", "1001"]


def test_sequential_number_handler_groups_on_a_composite_key():
    df = pd.DataFrame({"a": ["1", "1", "1"], "b": ["x", "x", "y"]})
    mapping = MappingRule(
        target_field="id",
        transformation={"type": "sequential_number", "assigned_per": "group", "start": 1},
    )
    context = TransformationContext(group_keys=["a", "b"])

    assert list(SequentialNumberHandler().apply(df, mapping, context)) == ["1", "1", "2"]


def test_sequential_number_handler_requires_a_declared_grouping():
    df = pd.DataFrame({"order": ["A"]})
    mapping = MappingRule(
        target_field="VBELN",
        transformation={"type": "sequential_number", "assigned_per": "group", "start": 1},
    )
    with pytest.raises(ValueError, match="no grouping is declared"):
        SequentialNumberHandler().apply(df, mapping)


def test_concatenate_handler_skip_empty_leaves_no_dangling_separator():
    df = pd.DataFrame({"a": ["shaft", "pipe"], "b": ["long", None]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="ARKTX",
        transformation={"type": "concatenate", "separator": " ", "skip_empty": True},
    )
    assert list(ConcatenateHandler().apply(df, mapping)) == ["shaft long", "pipe"]


def test_concatenate_handler_truncates_as_post_processing():
    df = pd.DataFrame({"a": ["x" * 30], "b": ["y" * 30]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="ARKTX",
        transformation={
            "type": "concatenate",
            "separator": " ",
            "skip_empty": True,
            "post_processing": {"type": "truncate", "length": 40},
        },
    )
    assert len(ConcatenateHandler().apply(df, mapping).iloc[0]) == 40


def test_lookup_handler_resolves_a_composite_key():
    """One legacy customer can hold several addresses that became separate
    customers, so both columns together identify the entry."""
    df = pd.DataFrame({"cust": ["1315011", "1315011"], "addr": ["2", "3"]})
    reference = pd.DataFrame(
        {
            "customer no. legacy": ["1315011", "1315011"],
            "address no.": ["2", "3"],
            "customer no. new": ["0000006765", "0120001914"],
        }
    )
    mapping = MappingRule(
        source_fields=["cust", "addr"],
        target_field="KUNNR",
        transformation={
            "type": "lookup",
            "reference": "customers",
            "lookup_key": ["customer no. legacy", "address no."],
            "return_field": "customer no. new",
        },
    )
    context = TransformationContext(reference_data={"customers": reference})

    result = LookupHandler().apply(df, mapping, context)

    assert list(result) == ["0000006765", "0120001914"]


def test_composite_lookup_does_not_collide_on_the_joined_key():
    """Regression: joining key parts on NUL made pandas' hashing treat
    "1315011<sep>2" and "1315011<sep>4" as one value, so an unambiguous table
    was rejected as ambiguous."""
    reference = pd.DataFrame(
        {
            "customer no. legacy": ["1315011"] * 3,
            "address no.": ["2", "3", "4"],
            "customer no. new": ["0000006765", "0120001914", "0000006765"],
        }
    )
    df = pd.DataFrame({"cust": ["1315011"], "addr": ["4"]})
    mapping = MappingRule(
        source_fields=["cust", "addr"],
        target_field="KUNNR",
        transformation={
            "type": "lookup",
            "reference": "customers",
            "lookup_key": ["customer no. legacy", "address no."],
            "return_field": "customer no. new",
        },
    )
    context = TransformationContext(reference_data={"customers": reference})

    assert list(LookupHandler().apply(df, mapping, context)) == ["0000006765"]


def _conversion_mapping(**overrides):
    transformation = {
        "type": "unit_conversion",
        "source_unit_field": "stock unit",
        "target_unit_field": "MEINS",
        "reference": "materials",
        "lookup_key": "part no.",
        "lookup_source_field": "part no.",
        "reference_fields": ["density"],
        "conversions": [
            {"from": "PCS", "to": "ST", "rule": "quantity"},
            {"from": "CM2", "to": "M2", "rule": "quantity / 10000"},
            {"from": "PCS", "to": "KG",
             "rule": "quantity * cross_section * length * density / 1000000"},
        ],
        "cross_section": {
            "field": "profile",
            "fields": ["width", "depth"],
            "rules": [
                {"profile": "round bar", "formula": "pi * (width / 2) ^ 2"},
                {"profile": "square bar", "formula": "width * depth"},
            ],
        },
    }
    transformation.update(overrides)
    return MappingRule(
        source_fields=["part no.", "quantity", "stock unit", "profile", "length", "width", "depth"],
        target_field="LBKUM",
        transformation=transformation,
    )


_CONVERSION_DF = pd.DataFrame(
    {
        "part no.": ["P1", "P2", "P3"],
        "quantity": ["29", "395", "48"],
        "stock unit": ["PCS", "CM2", "PCS"],
        "profile": [None, None, "round bar"],
        "length": [None, None, "500"],
        "width": [None, None, "8"],
        "depth": [None, None, None],
    }
)
_CONVERSION_REF = pd.DataFrame({"part no.": ["P1", "P2", "P3"], "density": ["7,85", "7,85", "1,14"]})


def _conversion_context(target_units):
    return TransformationContext(
        reference_data={"materials": _CONVERSION_REF},
        target_columns={"MEINS": pd.Series(target_units)},
    )


def test_unit_conversion_adopts_the_quantity_when_the_units_match():
    result = UnitConversionHandler().apply(
        _CONVERSION_DF, _conversion_mapping(), _conversion_context(["ST", "M2", "ST"])
    )
    assert result.iloc[0] == "29"


def test_unit_conversion_scales_between_related_units():
    result = UnitConversionHandler().apply(
        _CONVERSION_DF, _conversion_mapping(), _conversion_context(["ST", "M2", "ST"])
    )
    assert result.iloc[1] == "0,0395"


def test_unit_conversion_derives_a_weight_from_geometry_and_density():
    """48 round bars of 8 mm diameter and 500 mm length in polyamide:
    48 * pi * 4^2 * 500 * 1.14 / 1e6 kg."""
    mapping = _conversion_mapping(post_processing={"type": "round", "decimals": 3})
    result = UnitConversionHandler().apply(
        _CONVERSION_DF, mapping, _conversion_context(["ST", "M2", "KG"])
    )
    assert result.iloc[2] == "1,375"


def test_unit_conversion_rejects_an_uncovered_unit_pair():
    """A pair the table does not cover is an authoring error, not a silent
    pass-through."""
    with pytest.raises(ValueError, match="No conversion rule"):
        UnitConversionHandler().apply(
            _CONVERSION_DF, _conversion_mapping(), _conversion_context(["ST", "M2", "M"])
        )


def test_unit_conversion_defers_an_unresolved_target_unit():
    with pytest.raises(MissingTargetColumn):
        UnitConversionHandler().apply(
            _CONVERSION_DF, _conversion_mapping(), TransformationContext(
                reference_data={"materials": _CONVERSION_REF}
            )
        )


def test_calculation_handler_reads_already_built_target_columns():
    """A valuation price derived from a converted quantity and a price unit -
    neither of which is a source column."""
    df = pd.DataFrame({"stock value": ["860,14"]})
    mapping = MappingRule(
        source_fields=["stock value", "LBKUM", "PEINH"],
        target_field="VERPR",
        transformation={"type": "calculation", "formula": "stock value / LBKUM * PEINH",
                        "decimals": 2},
    )
    context = TransformationContext(
        target_columns={"LBKUM": pd.Series(["29,000"]), "PEINH": pd.Series(["1"])}
    )
    assert list(CalculationHandler().apply(df, mapping, context)) == ["29,66"]


def test_calculation_handler_defers_a_target_column_not_built_yet():
    df = pd.DataFrame({"stock value": ["860,14"]})
    mapping = MappingRule(
        source_fields=["stock value", "LBKUM"],
        target_field="VERPR",
        transformation={"type": "calculation", "formula": "stock value / LBKUM"},
    )
    with pytest.raises(MissingTargetColumn, match="LBKUM"):
        CalculationHandler().apply(df, mapping)


def test_to_numeric_strips_the_thousands_separator():
    """A German-locale export writes "4.028,000" - four thousand and
    twenty-eight, not a parse error (regression: the dot survived and the
    whole column became NaN)."""
    numbers, separator = _to_numeric(pd.Series(["4.028,000", "540,860", "30.975,200"]))

    assert list(numbers) == [4028.0, 540.86, 30975.2]
    assert separator == ","


def test_to_numeric_keeps_a_dot_decimal_when_no_comma_is_present():
    numbers, separator = _to_numeric(pd.Series(["2.5", "10"]))

    assert list(numbers) == [2.5, 10.0]
    assert separator == "."


def test_date_difference_handler_translates_a_due_date_into_a_payment_term():
    """The legacy system records only the due date; the target system expects
    the term it was derived from."""
    df = pd.DataFrame(
        {"Rechnungsdatum": ["12.04.22", "04.05.22"], "Zahlungstermin": ["26.04.22", "03.06.22"]}
    )
    mapping = MappingRule(
        source_fields=["Rechnungsdatum", "Zahlungstermin"],
        target_field="ZTERM",
        transformation={
            "type": "date_difference",
            "from_field": "Rechnungsdatum",
            "to_field": "Zahlungstermin",
            "input_format": "DD.MM.YY",
            "value_mapping": {14: "14 days net", 30: "30 days net"},
        },
    )
    result = DateDifferenceHandler().apply(df, mapping)
    assert list(result) == ["14 days net", "30 days net"]


def test_date_difference_handler_returns_the_day_count_without_a_mapping():
    df = pd.DataFrame({"a": ["12.04.22"], "b": ["26.04.22"]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="days",
        transformation={"type": "date_difference", "from_field": "a", "to_field": "b",
                        "input_format": "DD.MM.YY"},
    )
    assert list(DateDifferenceHandler().apply(df, mapping)) == ["14"]


def test_date_difference_handler_rejects_an_unmapped_difference():
    """An unexpected term must surface, not be emitted as a bare number."""
    df = pd.DataFrame({"a": ["12.04.22"], "b": ["19.04.22"]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="ZTERM",
        transformation={"type": "date_difference", "from_field": "a", "to_field": "b",
                        "input_format": "DD.MM.YY",
                        "value_mapping": {14: "14 days net"}},
    )
    with pytest.raises(ValueError, match="difference of"):
        DateDifferenceHandler().apply(df, mapping)

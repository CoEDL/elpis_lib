from pytest import raises

from elpis.models import ElanOptions, ElanTierSelector

VALID_ELAN_OPTIONS_DICT = {
    "selection_mechanism": "tier_name",
    "selection_value": "test",
}
INVALID_ELAN_OPTIONS_DICT = {
    "selection_mechanism": "pier_name",
    "selection_value": "jest",
}


def test_build_elan_options():
    options = ElanOptions.from_dict(VALID_ELAN_OPTIONS_DICT)
    assert options.selection_mechanism == ElanTierSelector.NAME
    assert options.selection_value == "test"


def test_build_invalid_elan_options_raises_error():
    with raises(ValueError):
        ElanOptions.from_dict(INVALID_ELAN_OPTIONS_DICT)


def test_serialize_elan_options():
    options = ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value="hello"
    )
    result = options.to_dict()
    assert result["selection_mechanism"] == "tier_name"
    assert result["selection_value"] == "hello"

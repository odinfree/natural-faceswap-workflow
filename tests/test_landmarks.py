import pytest

from natural_faceswap.landmarks import parse_point


def test_parse_normalized_point() -> None:
    assert parse_point("0.51,0.36") == (0.51, 0.36)


@pytest.mark.parametrize("value", ["2,0.2", "-0.1,0.2", "broken"])
def test_parse_point_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_point(value)

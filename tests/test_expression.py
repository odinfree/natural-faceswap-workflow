import pytest

from natural_faceswap.expression import ExpressionSettings


def test_default_expression_profile_is_valid() -> None:
    ExpressionSettings().validate()


def test_expression_alpha_must_be_bounded() -> None:
    with pytest.raises(ValueError, match="eye_core_alpha"):
        ExpressionSettings(eye_core_alpha=1.2).validate()


def test_expression_mode_must_be_known() -> None:
    with pytest.raises(ValueError, match="Unsupported expression mode"):
        ExpressionSettings(mode="face").validate()

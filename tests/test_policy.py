import pytest

from natural_faceswap.policy import PolicyError, RightsContext


def test_internal_evaluation_allows_unknown_source_and_model_rights() -> None:
    context = RightsContext(
        usage="internal-evaluation",
        source_rights="unknown",
        model_rights="unknown",
        identity_authorized=True,
    )
    context.validate()


def test_identity_authorization_is_always_required() -> None:
    context = RightsContext(
        usage="internal-evaluation",
        source_rights="confirmed",
        model_rights="confirmed",
        identity_authorized=False,
    )
    with pytest.raises(PolicyError, match="Identity authorization"):
        context.validate()


@pytest.mark.parametrize("field", ["source_rights", "model_rights"])
def test_commercial_requires_confirmed_rights(field: str) -> None:
    values = {
        "usage": "commercial",
        "source_rights": "confirmed",
        "model_rights": "confirmed",
        "identity_authorized": True,
    }
    values[field] = "unknown"
    with pytest.raises(PolicyError, match="Commercial use requires"):
        RightsContext(**values).validate()


def test_denied_rights_block_internal_evaluation() -> None:
    context = RightsContext(
        usage="internal-evaluation",
        source_rights="denied",
        model_rights="confirmed",
        identity_authorized=True,
    )
    with pytest.raises(PolicyError, match="Source rights are denied"):
        context.validate()

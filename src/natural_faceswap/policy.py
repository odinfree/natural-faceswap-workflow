from __future__ import annotations

from dataclasses import asdict, dataclass


class PolicyError(ValueError):
    """Raised when a render does not satisfy the declared rights gate."""


VALID_USAGE = {"internal-evaluation", "commercial"}
VALID_RIGHTS = {"unknown", "confirmed", "denied"}


@dataclass(frozen=True)
class RightsContext:
    usage: str
    source_rights: str
    model_rights: str
    identity_authorized: bool

    def validate(self) -> None:
        if self.usage not in VALID_USAGE:
            raise PolicyError(f"Invalid usage: {self.usage}")
        if self.source_rights not in VALID_RIGHTS:
            raise PolicyError(f"Invalid source rights: {self.source_rights}")
        if self.model_rights not in VALID_RIGHTS:
            raise PolicyError(f"Invalid model rights: {self.model_rights}")
        if not self.identity_authorized:
            raise PolicyError("Identity authorization is required.")
        if self.source_rights == "denied":
            raise PolicyError("Source rights are denied.")
        if self.model_rights == "denied":
            raise PolicyError("Model rights are denied.")
        if self.usage == "commercial":
            if self.source_rights != "confirmed":
                raise PolicyError("Commercial use requires confirmed source rights.")
            if self.model_rights != "confirmed":
                raise PolicyError("Commercial use requires confirmed model rights.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

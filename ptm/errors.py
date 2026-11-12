from typing import Any


class PTMError(Exception):
    def __init__(
        self, code: str, message: str, *, missing_fields: list[str] | None = None, status: int = 422
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.missing_fields = missing_fields or []
        self.status = status

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "missing_fields": self.missing_fields,
            "source_capability": "Capture-dependent; missing tensors, identities and denominators are never inferred",
            "retryable": False,
        }

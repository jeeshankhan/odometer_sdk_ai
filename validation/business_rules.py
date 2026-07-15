"""
Plausibility checks against a previous known reading. Odometers only go up
(barring rollover/tamper edge cases you may want to special-case), so a new
reading lower than the previous one, or implausibly higher, should be
flagged rather than silently trusted.
"""
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(str, Enum):
    VALID = "valid"
    REJECTED_LOWER_THAN_PREVIOUS = "rejected_lower_than_previous"
    REVIEW_REQUIRED_IMPLAUSIBLE_JUMP = "review_required_implausible_jump"
    VALID_NO_HISTORY = "valid_no_history"


@dataclass
class ValidationResult:
    status: ValidationStatus
    is_valid: bool
    message: str


def validate_reading(
    current_reading: int,
    previous_reading: int | None,
    max_plausible_daily_km: int = 1500,
    days_since_previous: float = 1.0,
) -> ValidationResult:
    if previous_reading is None:
        return ValidationResult(
            ValidationStatus.VALID_NO_HISTORY, True, "no previous reading to compare against"
        )

    if current_reading < previous_reading:
        return ValidationResult(
            ValidationStatus.REJECTED_LOWER_THAN_PREVIOUS,
            False,
            f"current reading {current_reading} is lower than previous {previous_reading}",
        )

    delta = current_reading - previous_reading
    max_plausible_delta = max_plausible_daily_km * max(days_since_previous, 1.0)

    if delta > max_plausible_delta:
        return ValidationResult(
            ValidationStatus.REVIEW_REQUIRED_IMPLAUSIBLE_JUMP,
            False,
            f"jump of {delta} exceeds plausible max of {max_plausible_delta:.0f} "
            f"over {days_since_previous:.1f} day(s)",
        )

    return ValidationResult(ValidationStatus.VALID, True, "within plausible range")

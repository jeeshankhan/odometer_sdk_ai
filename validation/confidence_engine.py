"""
Turns a raw RecognitionResult into an actionable confidence decision:
accept, retry-with-different-preprocessing, or ask-user-to-retake.
"""
from dataclasses import dataclass
from enum import Enum


class ConfidenceAction(str, Enum):
    ACCEPT = "accept"
    RETRY_PREPROCESSING = "retry_preprocessing"
    ASK_RETAKE = "ask_retake"


@dataclass
class ConfidenceDecision:
    action: ConfidenceAction
    reason: str
    low_confidence_digit_indices: list[int]


def evaluate_confidence(
    digit_confidences: list[float],
    accept_threshold: float = 0.97,
    retry_threshold: float = 0.80,
    max_low_digits_for_retry: int = 2,
) -> ConfidenceDecision:
    """
    digit_confidences: per-digit confidence in [0, 1].

    - All digits >= accept_threshold          -> ACCEPT
    - A small number of digits between        -> RETRY_PREPROCESSING
      retry_threshold and accept_threshold      (try different enhancement
                                                  settings, e.g. stronger CLAHE,
                                                  before bothering the user)
    - Many low digits, or any digit below      -> ASK_RETAKE
      retry_threshold
    """
    low_indices = [i for i, c in enumerate(digit_confidences) if c < accept_threshold]

    if not low_indices:
        return ConfidenceDecision(ConfidenceAction.ACCEPT, "all digits above accept threshold", [])

    very_low = [i for i in low_indices if digit_confidences[i] < retry_threshold]
    if very_low or len(low_indices) > max_low_digits_for_retry:
        return ConfidenceDecision(
            ConfidenceAction.ASK_RETAKE,
            f"{len(very_low)} digit(s) below retry threshold or too many uncertain digits",
            low_indices,
        )

    return ConfidenceDecision(
        ConfidenceAction.RETRY_PREPROCESSING,
        f"{len(low_indices)} digit(s) between {retry_threshold} and {accept_threshold}",
        low_indices,
    )

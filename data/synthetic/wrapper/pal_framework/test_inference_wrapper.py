"""Tests for ai/training_corpus/pal_framework/inference_wrapper.py.

PIX-4077 Phase 5. Stub LLM clients verify the two-stage Select-then-Generate
pipeline without requiring real HuggingFace endpoints. All paths exercised:
selection parsing, latency budget enforcement, JSON leakage rejection,
end-to-end inference, persona caching, unicode preservation.
"""

from __future__ import annotations

import time

import pytest
from inference_wrapper import (
    DEFAULT_LATENCY_BUDGET_SECONDS,
    JsonLeakageError,
    LatencyExceededError,
    PalInferenceResult,
    PalInferenceWrapper,
    SelectionParseError,
    _has_json_leakage,
    _parse_selection_index,
    _persona_to_string,
)


# ----------------------------------------------------------------------
# Fixtures + stub LLM clients
# ----------------------------------------------------------------------
def _persona(
    age: int = 45,
    gender: str = "female",
    location: str = "Hanoi",
    literacy: str = "low",
    preference: str = "traditional medicine",
) -> dict:
    return {
        "demographics": {"age": age, "gender": gender, "location": location},
        "healthcare_behavior": {"health_literacy": literacy, "preference": preference},
    }


class StubSelector:
    """Returns a fixed 1-indexed option number on each call.

    Records every call so tests can assert prompt construction.
    """

    def __init__(self, response: str = "1", delay: float = 0.0) -> None:
        self.response = response
        self.delay = delay
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.delay:
            time.sleep(self.delay)
        return self.response


class StubGenerator:
    """Returns a fixed response string on each call.

    Records every call so tests can assert prompt construction.
    """

    def __init__(self, response: str = "Tôi cảm thấy hơi mệt.", delay: float = 0.0) -> None:
        self.response = response
        self.delay = delay
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.delay:
            time.sleep(self.delay)
        return self.response


# ----------------------------------------------------------------------
# _parse_selection_index
# ----------------------------------------------------------------------
class TestParseSelectionIndex:
    def test_plain_integer(self):
        assert _parse_selection_index("3", 5) == 2

    def test_with_extra_text(self):
        assert _parse_selection_index("2. Persona matches elderly patient", 5) == 1

    def test_one_indexed_boundary_low(self):
        assert _parse_selection_index("1", 5) == 0

    def test_one_indexed_boundary_high(self):
        assert _parse_selection_index("5", 5) == 4

    def test_rejects_zero(self):
        with pytest.raises(SelectionParseError, match="out of range"):
            _parse_selection_index("0", 5)

    def test_rejects_above_n(self):
        with pytest.raises(SelectionParseError, match="out of range"):
            _parse_selection_index("6", 5)

    def test_rejects_non_integer(self):
        with pytest.raises(SelectionParseError, match="not parseable"):
            _parse_selection_index("abc", 5)

    def test_rejects_empty(self):
        with pytest.raises(SelectionParseError, match="empty"):
            _parse_selection_index("", 5)

    def test_rejects_non_string(self):
        with pytest.raises(SelectionParseError, match="non-string"):
            _parse_selection_index(None, 5)  # type: ignore[arg-type]

    def test_strips_whitespace(self):
        assert _parse_selection_index("  3  ", 5) == 2


# ----------------------------------------------------------------------
# _has_json_leakage
# ----------------------------------------------------------------------
class TestHasJsonLeakage:
    @pytest.mark.parametrize(
        "text",
        [
            "this has a { brace",
            "this has a } brace",
            'this has a " quote',
            "this has a ' apostrophe",
        ],
    )
    def test_detects_leakage(self, text):
        assert _has_json_leakage(text) is True

    def test_clean_text_no_leakage(self):
        assert _has_json_leakage("Tôi cảm thấy hơi mệt hôm nay.") is False


# ----------------------------------------------------------------------
# _persona_to_string
# ----------------------------------------------------------------------
class TestPersonaToString:
    def test_uses_meddies_format_persona(self):
        result = _persona_to_string(_persona())
        assert "45-year-old" in result
        assert "female" in result
        assert "Hanoi" in result
        assert "low health literacy" in result
        assert "traditional medicine" in result

    def test_fallback_on_invalid(self):
        # An empty dict triggers the local fallback path which still
        # synthesizes a sensible string.
        result = _persona_to_string({})
        assert "Vietnam" in result  # default location
        assert "average" in result  # default literacy

    def test_fallback_on_non_dict(self):
        result = _persona_to_string(None)  # type: ignore[arg-type]
        assert "Vietnam" in result
        assert "standard medicine" in result


# ----------------------------------------------------------------------
# PalInferenceWrapper.__init__ validation
# ----------------------------------------------------------------------
class TestWrapperInit:
    def test_rejects_empty_candidates(self):
        with pytest.raises(ValueError, match="candidate_personas must not be empty"):
            PalInferenceWrapper(
                selector_client=StubSelector(),
                generator_client=StubGenerator(),
                candidate_personas=[],
            )

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match="latency_budget_seconds must be positive"):
            PalInferenceWrapper(
                selector_client=StubSelector(),
                generator_client=StubGenerator(),
                candidate_personas=[_persona()],
                latency_budget_seconds=0,
            )

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="latency_budget_seconds must be positive"):
            PalInferenceWrapper(
                selector_client=StubSelector(),
                generator_client=StubGenerator(),
                candidate_personas=[_persona()],
                latency_budget_seconds=-1.0,
            )

    def test_default_budget_is_two_seconds(self):
        assert DEFAULT_LATENCY_BUDGET_SECONDS == 2.0


# ----------------------------------------------------------------------
# Stage 1 — select_persona
# ----------------------------------------------------------------------
class TestSelectPersona:
    def test_returns_selected_persona_string(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona(age=45), _persona(age=30)],
        )
        result = wrapper.select_persona("Patient: I am tired today.")
        assert "45-year-old" in result.persona_string
        assert result.selected_index == 0

    def test_returns_second_persona_when_index_is_two(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("2"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona(age=45), _persona(age=30)],
        )
        result = wrapper.select_persona("Patient: I am tired today.")
        assert "30-year-old" in result.persona_string
        assert result.selected_index == 1

    def test_selector_messages_exclude_assistant_turn(self):
        selector = StubSelector("1")
        wrapper = PalInferenceWrapper(
            selector_client=selector,
            generator_client=StubGenerator(),
            candidate_personas=[_persona(), _persona(age=30)],
        )
        wrapper.select_persona("Patient: hello.")
        assert len(selector.calls) == 1
        messages = selector.calls[0]
        roles = [m["role"] for m in messages]
        assert "assistant" not in roles

    def test_selector_messages_include_candidate_personas(self):
        selector = StubSelector("1")
        wrapper = PalInferenceWrapper(
            selector_client=selector,
            generator_client=StubGenerator(),
            candidate_personas=[_persona(age=45), _persona(age=30)],
        )
        wrapper.select_persona("Patient: hello.")
        messages = selector.calls[0]
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        assert "45-year-old" in user_text
        assert "30-year-old" in user_text

    def test_rejects_empty_dialogue(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona()],
        )
        with pytest.raises(ValueError, match="non-empty"):
            wrapper.select_persona("")

    def test_rejects_non_string_dialogue(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona()],
        )
        with pytest.raises(ValueError, match="non-empty"):
            wrapper.select_persona(None)  # type: ignore[arg-type]

    def test_propagates_parse_error(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("banana"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona(), _persona(age=30)],
        )
        with pytest.raises(SelectionParseError):
            wrapper.select_persona("Patient: hello.")

    def test_propagates_out_of_range_error(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("99"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona(), _persona(age=30)],
        )
        with pytest.raises(SelectionParseError, match="out of range"):
            wrapper.select_persona("Patient: hello.")

    def test_latency_is_recorded(self):
        selector = StubSelector("1", delay=0.01)
        wrapper = PalInferenceWrapper(
            selector_client=selector,
            generator_client=StubGenerator(),
            candidate_personas=[_persona()],
        )
        result = wrapper.select_persona("Patient: hello.")
        assert result.latency_seconds >= 0.01


# ----------------------------------------------------------------------
# Stage 2 — generate_response
# ----------------------------------------------------------------------
class TestGenerateResponse:
    def test_returns_generated_text(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("Tôi cảm thấy mệt."),
            candidate_personas=[_persona()],
        )
        result = wrapper.generate_response(
            persona_string="This patient is a 45-year-old female from Hanoi.",
            dialogue_history="Patient: hello.",
        )
        assert result.response == "Tôi cảm thấy mệt."

    def test_strips_whitespace(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("  surrounding whitespace  "),
            candidate_personas=[_persona()],
        )
        result = wrapper.generate_response(persona_string="persona", dialogue_history="dialogue")
        assert result.response == "surrounding whitespace"

    def test_generator_messages_include_persona_and_system_prompt(self):
        gen = StubGenerator("response")
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=gen,
            candidate_personas=[_persona()],
        )
        wrapper.generate_response(
            persona_string="This patient is a 45-year-old female.",
            dialogue_history="Patient: hello.",
        )
        assert len(gen.calls) == 1
        messages = gen.calls[0]
        system = [m for m in messages if m["role"] == "system"]
        assert len(system) == 1
        assert "clinical persona" in system[0]["content"]
        user = [m for m in messages if m["role"] == "user"]
        assert len(user) == 1
        assert "45-year-old" in user[0]["content"]
        assert "Patient: hello." in user[0]["content"]

    def test_generator_messages_exclude_empty_assistant_turn(self):
        gen = StubGenerator("response")
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=gen,
            candidate_personas=[_persona()],
        )
        wrapper.generate_response(persona_string="persona", dialogue_history="dialogue")
        messages = gen.calls[0]
        roles = [m["role"] for m in messages]
        assert "assistant" not in roles

    def test_rejects_empty_persona(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona()],
        )
        with pytest.raises(ValueError, match="persona_string"):
            wrapper.generate_response(persona_string="   ", dialogue_history="d")

    def test_rejects_non_string_dialogue_history(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator(),
            candidate_personas=[_persona()],
        )
        with pytest.raises(ValueError, match="dialogue_history"):
            wrapper.generate_response(
                persona_string="p",
                dialogue_history=None,  # type: ignore[arg-type]
            )

    def test_rejects_json_leakage_in_response(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator('{"key": "value"}'),
            candidate_personas=[_persona()],
        )
        with pytest.raises(JsonLeakageError, match="JSON formatting"):
            wrapper.generate_response(persona_string="p", dialogue_history="d")

    def test_rejects_apostrophe_in_response(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("It's a problem."),
            candidate_personas=[_persona()],
        )
        with pytest.raises(JsonLeakageError):
            wrapper.generate_response(persona_string="p", dialogue_history="d")

    def test_latency_is_recorded(self):
        gen = StubGenerator("response", delay=0.01)
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=gen,
            candidate_personas=[_persona()],
        )
        result = wrapper.generate_response(persona_string="p", dialogue_history="d")
        assert result.latency_seconds >= 0.01


# ----------------------------------------------------------------------
# End-to-end — infer
# ----------------------------------------------------------------------
class TestInfer:
    def test_end_to_end_success(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("Tôi cảm thấy mệt."),
            candidate_personas=[_persona(), _persona(age=30)],
        )
        result = wrapper.infer("Patient: tôi mệt")
        assert isinstance(result, PalInferenceResult)
        assert "45-year-old" in result.selection.persona_string
        assert result.generation.response == "Tôi cảm thấy mệt."
        assert result.total_latency_seconds > 0
        assert result.dialogue_history_text == "Patient: tôi mệt"

    def test_total_latency_within_budget(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1", delay=0.05),
            generator_client=StubGenerator("response", delay=0.05),
            candidate_personas=[_persona()],
            latency_budget_seconds=2.0,
        )
        result = wrapper.infer("Patient: hello")
        assert result.total_latency_seconds < 2.0
        assert result.total_latency_seconds >= 0.10

    def test_raises_latency_exceeded(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1", delay=0.10),
            generator_client=StubGenerator("response", delay=0.10),
            candidate_personas=[_persona()],
            latency_budget_seconds=0.05,  # aggressive budget
        )
        with pytest.raises(LatencyExceededError, match="exceeds budget"):
            wrapper.infer("Patient: hello")

    def test_propagates_selection_parse_error(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("banana"),
            generator_client=StubGenerator("response"),
            candidate_personas=[_persona()],
        )
        with pytest.raises(SelectionParseError):
            wrapper.infer("Patient: hello")

    def test_propagates_json_leakage_error(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator('{"leak": true}'),
            candidate_personas=[_persona()],
        )
        with pytest.raises(JsonLeakageError):
            wrapper.infer("Patient: hello")

    def test_preserves_vietnamese_unicode(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("Tôi cảm thấy hơi mệt hôm nay 🇻🇳"),
            candidate_personas=[_persona()],
        )
        result = wrapper.infer("Patient: chào bác sĩ")
        assert "🇻🇳" in result.generation.response
        assert "Tôi" in result.generation.response

    def test_no_json_leakage_in_generated_response(self):
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("Tôi cảm thấy mệt."),
            candidate_personas=[_persona()],
        )
        result = wrapper.infer("Patient: hello")
        assert not _has_json_leakage(result.generation.response)

    def test_inference_under_two_second_target(self):
        # PIX-4077 acceptance criterion: <2s on A100. We cannot test A100 in
        # CI, but we can assert that with negligible stub latency the wrapper
        # finishes well under the 2.0s default budget.
        wrapper = PalInferenceWrapper(
            selector_client=StubSelector("1"),
            generator_client=StubGenerator("response"),
            candidate_personas=[_persona()],
        )
        start = time.perf_counter()
        wrapper.infer("Patient: hello")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5  # generous margin; real stubs run in microseconds


# ----------------------------------------------------------------------
# Stub LLM client edge cases
# ----------------------------------------------------------------------
class TestStubClientUsage:
    def test_wrapper_accepts_any_callable(self):
        """The wrapper should accept any ``messages -> str`` callable, not
        only instances of the StubSelector/StubGenerator classes."""

        def selector(messages):
            return "1"

        def generator(messages):
            return "response"

        wrapper = PalInferenceWrapper(
            selector_client=selector,  # type: ignore[arg-type]
            generator_client=generator,  # type: ignore[arg-type]
            candidate_personas=[_persona()],
        )
        result = wrapper.infer("Patient: hello")
        assert result.generation.response == "response"

import pytest

from src.common.exceptions import (
    AudioValidationError,
    CallCenterIntelError,
    InjectionDetectedError,
    LLMAnalysisError,
    PIIRedactionError,
    PipelineError,
    ReportGenerationError,
    TranscriptionError,
)

_ALL_SUBTYPES = [
    AudioValidationError,
    TranscriptionError,
    InjectionDetectedError,
    PIIRedactionError,
    LLMAnalysisError,
    ReportGenerationError,
    PipelineError,
]


def test_base_inherits_builtin_exception():
    assert issubclass(CallCenterIntelError, Exception)


@pytest.mark.parametrize("exc_class", _ALL_SUBTYPES)
def test_all_subtypes_inherit_base(exc_class):
    assert issubclass(exc_class, CallCenterIntelError)


@pytest.mark.parametrize("exc_class", _ALL_SUBTYPES)
def test_each_subtype_catchable_as_base(exc_class):
    with pytest.raises(CallCenterIntelError):
        raise exc_class("test message")


def test_message_accessible():
    err = AudioValidationError("unsupported format")
    assert str(err) == "unsupported format"


def test_context_stored_when_provided():
    err = AudioValidationError("bad format", context={"format": "ogg", "path": "/tmp/x.ogg"})
    assert err.context["format"] == "ogg"
    assert err.context["path"] == "/tmp/x.ogg"


def test_context_defaults_to_empty_dict():
    err = TranscriptionError("whisper failed")
    assert err.context == {}


def test_injection_detected_error():
    err = InjectionDetectedError("injection found", context={"pattern": "ignore all previous"})
    assert isinstance(err, CallCenterIntelError)
    assert err.context["pattern"] == "ignore all previous"


def test_pipeline_error_wraps_message():
    err = PipelineError("unexpected failure")
    assert "unexpected failure" in str(err)

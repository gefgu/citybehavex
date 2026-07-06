from __future__ import annotations

import json

import pytest

from citybehavex.llm import LLMConfig
from citybehavex.llm_diaries import DiariesConfig, DiaryValidationError, LLMStats
from citybehavex.profiles import AgentProfilesConfig, WEIGHT_GROUPS, calibrate_demographic_weights
from citybehavex.simulation.runner import resolve_calibrated_profiles_config


class _Response:
    text = ""

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _chat(payload: dict) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}


def _weights_payload(scale: float = 1.0) -> dict:
    return {field: [scale] * len(labels) for field, labels in WEIGHT_GROUPS.items()}


def _config(**overrides) -> LLMConfig:
    defaults = dict(base_url="http://localhost:8000", api_key="test", model="test-model")
    defaults.update(overrides)
    return LLMConfig(**defaults)


def test_returns_none_when_no_llm_configured():
    result = calibrate_demographic_weights(LLMConfig(), city_profile="a small town")
    assert result is None


def test_calibrates_and_normalizes_all_groups():
    calls = []

    def do_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return _Response(_chat(_weights_payload()))

    class FakeRequests:
        get = staticmethod(lambda *a, **k: _Response({"data": []}))
        post = staticmethod(do_post)

    stats = LLMStats()
    result = calibrate_demographic_weights(
        _config(), city_profile="a wealthy university city", stats=stats, requests_module=FakeRequests
    )

    assert result is not None
    assert set(result) == set(WEIGHT_GROUPS)
    for field, labels in WEIGHT_GROUPS.items():
        assert len(result[field]) == len(labels)
        assert result[field] == pytest.approx([1.0 / len(labels)] * len(labels))
    assert stats.calls == 1
    assert len(calls) == 1


def test_raises_on_missing_group():
    payload = _weights_payload()
    del payload["job_weights"]

    class FakeRequests:
        get = staticmethod(lambda *a, **k: _Response({"data": []}))
        post = staticmethod(lambda *a, **k: _Response(_chat(payload)))

    with pytest.raises(DiaryValidationError, match="job_weights"):
        calibrate_demographic_weights(
            _config(retries=1), city_profile="test city", requests_module=FakeRequests
        )


def test_raises_on_wrong_length():
    payload = _weights_payload()
    payload["education_weights"] = [1.0, 1.0]

    class FakeRequests:
        get = staticmethod(lambda *a, **k: _Response({"data": []}))
        post = staticmethod(lambda *a, **k: _Response(_chat(payload)))

    with pytest.raises(DiaryValidationError, match="education_weights"):
        calibrate_demographic_weights(
            _config(retries=1), city_profile="test city", requests_module=FakeRequests
        )


def test_raises_on_negative_weight():
    payload = _weights_payload()
    payload["health_weights"][0] = -1.0

    class FakeRequests:
        get = staticmethod(lambda *a, **k: _Response({"data": []}))
        post = staticmethod(lambda *a, **k: _Response(_chat(payload)))

    with pytest.raises(DiaryValidationError, match="non-negative"):
        calibrate_demographic_weights(
            _config(retries=1), city_profile="test city", requests_module=FakeRequests
        )


def test_resolve_falls_back_to_defaults_when_llm_override_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(
        "citybehavex.simulation.runner.calibrate_demographic_weights",
        lambda *a, **k: called.append(1) or _weights_payload(0.5),
    )
    pc = AgentProfilesConfig(llm_override=False)
    result = resolve_calibrated_profiles_config(pc, LLMConfig(), DiariesConfig())
    assert result is pc
    assert not called


def test_resolve_applies_calibrated_weights(monkeypatch):
    monkeypatch.setattr(
        "citybehavex.simulation.runner.calibrate_demographic_weights",
        lambda *a, **k: _weights_payload(0.5),
    )
    pc = AgentProfilesConfig(llm_override=True)
    result = resolve_calibrated_profiles_config(
        pc, LLMConfig(), DiariesConfig(city_profile="test city")
    )
    assert result is not pc
    for field, labels in WEIGHT_GROUPS.items():
        assert getattr(result, field) == pytest.approx([0.5] * len(labels))


def test_resolve_falls_back_to_defaults_after_calibration_failure(monkeypatch, capsys):
    def raise_failure(*args, **kwargs):
        raise DiaryValidationError("LLM demographic weight calibration failed after 3 attempt(s)")

    monkeypatch.setattr(
        "citybehavex.simulation.runner.calibrate_demographic_weights", raise_failure
    )
    pc = AgentProfilesConfig(llm_override=True)
    result = resolve_calibrated_profiles_config(
        pc, LLMConfig(), DiariesConfig(city_profile="test city")
    )
    assert result == pc
    assert "using default weights" in capsys.readouterr().out


def test_resolve_falls_back_when_no_llm_configured(monkeypatch):
    monkeypatch.setattr(
        "citybehavex.simulation.runner.calibrate_demographic_weights", lambda *a, **k: None
    )
    pc = AgentProfilesConfig(llm_override=True)
    result = resolve_calibrated_profiles_config(pc, LLMConfig(), DiariesConfig())
    assert result is pc

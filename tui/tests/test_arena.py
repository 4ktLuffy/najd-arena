from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidTag

from najd_arena.adapters import adapt, supported_sources
from najd_arena.hosted import CredentialCipher, EncryptedSecret
from najd_arena.metrics import aggregate, deterministic_grade
from najd_arena.models import BenchmarkCase


def case(source: str, expected: dict, *, prompt: str = "question") -> BenchmarkCase:
    return BenchmarkCase("one", "arabic", "ar", prompt, (), expected,
                         {"sourceId": source}, ())


def test_registry_covers_every_release_source() -> None:
    assert len(supported_sources()) == 39


def test_multiple_choice_adapter_and_grade() -> None:
    value = case("arabicmmlu", {"answerKey": "B", "options": ["a", "b"]})
    assert adapt(value).mode == "exact"
    assert adapt(value).messages[1]["content"] == "question\n\nA. a\nB. b"
    assert deterministic_grade(value, "B") == {
        "method": "arabic-mmlu-key-v1", "score": 1.0, "passed": True,
        "parsed_key": "B", "valid_output": True,
    }
    assert deterministic_grade(value, "B\nBecause...")["score"] == 0.0


def test_tool_call_grading() -> None:
    value = case("arabic-function-calling", {
        "functionName": "weather", "arguments": {"city": "الرياض"}, "requiresFunction": True,
    })
    output = '{"functionName":"weather","arguments":{"city":"الرياض"}}'
    assert deterministic_grade(value, output)["score"] == 1


@pytest.mark.parametrize("source", ["arabic-function-calling", "arabic-agent-eval", "paired-msa-saudi-tool-use"])
@pytest.mark.parametrize("output", [
    "5", '"x"', "true", "null", "[]", '[{"function":"f","arguments":{}}]', "see [1] for details",
    "not json at all", "",
])
def test_tool_grading_scores_non_object_output_zero(source: str, output: str) -> None:
    value = case(source, {
        "functionName": "weather", "arguments": {}, "tool_calls": [], "action": "call_tool",
        "tool_name": "weather", "missing_required_arguments": [], "gold_call": None,
    })
    assert deterministic_grade(value, output)["score"] == 0


def test_open_answer_requires_judge() -> None:
    value = case("QCRI/IslamicFaithQA", {"answer": "يونس"})
    assert adapt(value).mode == "judge"
    assert deterministic_grade(value, "النبي يونس") is None


def test_mena_distribution_score_is_normalized() -> None:
    value = case("mena-values", {"min": 1.0, "max": 4.0, "scale": "y",
        "surveyDistributions": {"Saudi": "(1.3, {'1': '70%', '2': '20%', '3': '7%', '4': '3%'})"}})
    assert deterministic_grade(value, "1")["score"] == 1


def test_macro_track_aggregation() -> None:
    report = aggregate([
        {"track": "large", "score": 1.0}, {"track": "large", "score": 1.0},
        {"track": "small", "score": 0.0},
    ])
    assert report["najd_score"] == 0.5
    assert report["case_weighted_score"] == pytest.approx(2 / 3, abs=1e-6)


def test_credentials_round_trip_with_authenticated_context() -> None:
    cipher = CredentialCipher(b"x" * 32)
    encrypted = cipher.encrypt("secret", run_id="run", organization_id="org")
    assert cipher.decrypt(encrypted, run_id="run", organization_id="org") == "secret"
    with pytest.raises(InvalidTag):
        cipher.decrypt(encrypted, run_id="other", organization_id="org")
    web_style = EncryptedSecret(encrypted.ciphertext, encrypted.nonce, encrypted.auth_tag,
                                encrypted.key_id)
    assert cipher.decrypt(web_style, run_id="run", organization_id="org") == "secret"


def test_environment_key_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_CREDENTIAL_KEY", base64.urlsafe_b64encode(b"z" * 32).decode())
    assert CredentialCipher.from_environment().key_id == "v1"

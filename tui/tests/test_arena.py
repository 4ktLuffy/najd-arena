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


def test_open_answer_requires_judge() -> None:
    value = case("QCRI/IslamicFaithQA", {"answer": "يونس"})
    assert adapt(value).mode == "judge"
    assert deterministic_grade(value, "النبي يونس") is None


def test_mena_distribution_score_is_normalized() -> None:
    value = case("mena-values", {"min": 1.0, "max": 4.0, "scale": "y",
        "surveyDistributions": {"Saudi": "(1.3, {'1': '70%', '2': '20%', '3': '7%', '4': '3%'})"}})
    assert deterministic_grade(value, "1")["score"] == 1


def mena_case(*distributions: str) -> BenchmarkCase:
    return case("mena-values", {"min": 1.0, "max": 4.0, "scale": "y", "surveyDistributions": {
        f"Country{index}": text for index, text in enumerate(distributions)}})


def test_mena_float_style_option_keys_match_integer_answer() -> None:
    value = mena_case("(2.1, {'1.0': '10%', '2.0': '60%', '3.0': '30%'})")
    assert deterministic_grade(value, "2")["score"] == 1
    assert deterministic_grade(value, "3")["score"] == pytest.approx(0.5)
    assert deterministic_grade(value, "2.0")["score"] == 1
    assert deterministic_grade(value, "9")["score"] == 0


def test_mena_option_key_styles_are_averaged_together() -> None:
    value = mena_case("(2.0, {'2': '40%', '3': '60%'})", "(2.0, {2.0: '80%', 3.0: '20%'})",
                      "(2.0, {'2.0': '60%', '3.0': '10%'})")
    assert deterministic_grade(value, "2")["score"] == pytest.approx(1)
    assert deterministic_grade(value, "3")["score"] == pytest.approx((90 / 3) / (180 / 3))


def test_mena_non_numeric_option_keys_still_work() -> None:
    value = mena_case("(1.0, {'agree': '70%', '1': '30%'})")
    assert deterministic_grade(value, "1")["score"] == pytest.approx(30 / 70)
    assert deterministic_grade(value, "agree")["score"] == 0


def test_mena_without_usable_distribution_scores_zero() -> None:
    value = mena_case("(No data for this question)")
    assert deterministic_grade(value, "1")["score"] == 0


def test_mena_huge_number_scores_zero_instead_of_raising() -> None:
    value = mena_case("(2.0, {'1.0': '50%', '2.0': '50%'})")
    assert deterministic_grade(value, "9" * 400)["score"] == 0
    assert deterministic_grade(value, "-" + "9" * 400)["score"] == 0


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

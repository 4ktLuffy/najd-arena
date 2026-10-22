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


def test_options_are_appended_with_latin_letters() -> None:
    value = case("alghafa-native", {"answerKey": "1", "options": ["red", "green", "blue"]}, prompt="Pick one")
    adapted = adapt(value)
    assert adapted.mode == "exact"
    assert adapted.messages[1]["content"] == "Pick one\n\nA. red\nB. green\nC. blue"
    assert "letter" in adapted.messages[0]["content"]
    assert "index" not in adapted.messages[0]["content"]


def test_embedded_position_labels_are_stripped_once() -> None:
    value = case("mixed", {"answerKey": "B", "options": ["A. one", "(B) two", "ج) three", "d- four", "E five"]})
    lines = adapt(value).messages[1]["content"].splitlines()[2:]
    assert lines == ["A. one", "B. two", "C. three", "D. four", "E. E five"]
    # a label for another position is data, not a label for this one
    other = case("mixed", {"answerKey": "A", "options": ["B. one", "A. two"]})
    assert adapt(other).messages[1]["content"].splitlines()[2:] == ["A. B. one", "B. A. two"]


def test_options_already_listed_in_prompt_are_not_repeated() -> None:
    value = case("mixed", {"answerKey": "A", "options": ["red", "green"]}, prompt="Pick one\nA. red\nB) green")
    assert adapt(value).messages[1]["content"] == "Pick one\nA. red\nB) green"
    # option words inside running prose are not a list
    prose = case("mixed", {"answerKey": "A", "options": ["red", "green"]}, prompt="red and green are colours")
    assert adapt(prose).messages[1]["content"].endswith("A. red\nB. green")


def test_arabic_letter_keys() -> None:
    value = case("aratrust", {"answerKey": "ب", "options": ["أ) x", "ب) y", "ج) z"]})
    assert adapt(value).messages[1]["content"].endswith("A. x\nB. y\nC. z")
    for output, score in [("B", 1), ("ب", 1), ("b.", 1), ("y", 1), ("ب) y", 1), ("A", 0), ("أ", 0), ("x", 0)]:
        assert deterministic_grade(value, output)["score"] == score, output


def test_digit_string_keys_are_zero_based() -> None:
    value = case("alghafa-native", {"answerKey": "0", "options": ["first", "second", "third"]})
    assert deterministic_grade(value, "A")["score"] == 1
    assert deterministic_grade(value, "B")["score"] == 0
    assert deterministic_grade(value, "first")["score"] == 1
    last = case("alghafa-native", {"answerKey": "2", "options": ["first", "second", "third"]})
    assert deterministic_grade(last, "C")["score"] == 1
    assert deterministic_grade(last, "third")["score"] == 1
    assert deterministic_grade(last, "B")["score"] == 0


def test_answer_index_is_one_based() -> None:
    value = case("humain-arapro", {"answerIndex": 3, "options": ["w", "x", "y", "z"]})
    assert deterministic_grade(value, "C")["score"] == 1
    assert deterministic_grade(value, "y")["score"] == 1
    # the next option's letter or text used to score 1.0 through the 0-based text fallback
    assert deterministic_grade(value, "D")["score"] == 0
    assert deterministic_grade(value, "z")["score"] == 0
    assert deterministic_grade(value, '{"answerIndex": 3}')["score"] == 1


def test_option_text_needs_equality_not_substring() -> None:
    value = case("dialectal-arabic-mmlu", {"answerKey": "0", "options": ["0.2", "0.25", "0.3"]})
    assert deterministic_grade(value, "0.2")["score"] == 1
    assert deterministic_grade(value, "A. 0.2")["score"] == 1
    assert deterministic_grade(value, "0.25")["score"] == 0
    assert deterministic_grade(value, "the answer is 0.2")["score"] == 0
    assert deterministic_grade(value, "B. 0.2")["score"] == 0
    longer = case("dialectal-arabic-mmlu", {"answerKey": "1", "options": ["0.2", "0.25", "0.3"]})
    assert deterministic_grade(longer, "0.2")["score"] == 0
    assert deterministic_grade(longer, "0.25")["score"] == 1


def test_choice_json_keys_and_method() -> None:
    value = case("humain-aratruthfulqa", {"answerKey": "C", "options": ["A. p", "B. q", "C. r"]})
    assert deterministic_grade(value, '{"answerKey": "C"}') == {
        "method": "exact-choice@2", "score": 1.0, "passed": True,
    }
    assert deterministic_grade(value, '{"answerKey": "B"}')["score"] == 0
    assert deterministic_grade(value, "r")["score"] == 1
    assert deterministic_grade(value, "")["score"] == 0


def test_case_without_options_keeps_old_behaviour() -> None:
    value = case("commonsense-validation", {"answerKey": "1"}, prompt="question")
    assert adapt(value).messages[1]["content"] == "question"
    assert "index" in adapt(value).messages[0]["content"]
    assert deterministic_grade(value, "1") == {"method": "exact@1", "score": 1.0, "passed": True}
    assert deterministic_grade(value, "0")["score"] == 0


def test_absher_correction_path_is_unchanged() -> None:
    value = BenchmarkCase("one", "arabic", "ar", "question", (), {"answer": "ب", "options": ["x", "y"]},
                          {"sourceId": "absher", "correction_version": "2026.09.27.1"}, ())
    adapted = adapt(value)
    assert adapted.messages[1]["content"] == "question"
    assert "Arabic option key" in adapted.messages[0]["content"]


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

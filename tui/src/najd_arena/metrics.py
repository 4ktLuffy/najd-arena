from __future__ import annotations

import ast
import json
import re
import unicodedata
from statistics import mean
from typing import Any

from najd_benchmark.arabic_mmlu import grade as grade_arabic_mmlu

from .adapters import adapt, json_from_output, mcq_options, normalize_token, option_texts, strip_option_label
from .models import BenchmarkCase

CHOICE_METHOD = "exact-choice@2"
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _text_key(value: Any) -> str:
    """Normalise option text for equality; unlike normalize_token it keeps inner punctuation ("0.25" != "0.2")."""
    text = unicodedata.normalize("NFKC", str(value)).translate(_ARABIC_DIGITS)
    text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,;:!?\"'()[]*_`،؛").casefold()


def _gold_position(expected: dict[str, Any], count: int) -> int | None:
    """0-based position of the correct option.

    answerKey is a letter (Latin or Arabic) or, in alghafa-native and dialectal-arabic-mmlu, a 0-based digit
    string. answerIndex (humain-arapro) is 1-based.
    """
    key = expected.get("answerKey")
    index = expected.get("answerIndex")
    position: int | None = None
    if isinstance(key, str):
        token = normalize_token(key)
        if len(token) == 1 and "a" <= token <= "z":
            position = ord(token) - ord("a")
        elif token.isdigit():
            position = int(token)
    elif key is None and isinstance(index, int) and not isinstance(index, bool):
        position = index - 1
    return position if position is not None and 0 <= position < count else None


def _choice(case: BenchmarkCase, output: str) -> float:
    expected = case.expected
    wanted: Any = expected.get("answerKey", expected.get("answerIndex"))
    candidate: Any = output.strip().splitlines()[0] if output.strip() else ""
    from_json = False
    try:
        parsed = json_from_output(output)
        if isinstance(parsed, dict):
            candidate = parsed.get("answerKey", parsed.get("answerIndex", candidate))
            from_json = True
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    options = mcq_options(case)
    position = _gold_position(expected, len(options)) if options else None
    if options is None or position is None:
        return float(normalize_token(candidate) == normalize_token(wanted))
    if from_json and normalize_token(candidate) == normalize_token(wanted):
        return 1.0  # {"answerKey": ...} / {"answerIndex": ...} echoing the stored key or index
    token = normalize_token(candidate)
    if len(token) == 1 and "a" <= token <= "z" and ord(token) - ord("a") == position:
        return 1.0
    # Option text, optionally preceded by its own letter. A different letter stays in the text and cannot match.
    gold = _text_key(option_texts(options)[position])
    # The whole output is also tried, for options whose own text spans several lines.
    return float(any(_text_key(strip_option_label(str(text), position)) == gold for text in (candidate, output)))


def _structured(case: BenchmarkCase, output: str) -> float:
    expected = case.expected
    try:
        parsed = json_from_output(output)
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    keys = [key for key in expected if key not in {"options", "answer", "numericAnswer"}]
    if not keys:
        return float(normalize_token(output) == normalize_token(expected.get("answer", "")))
    return mean(float(parsed.get(key) == expected[key]) for key in keys)


def _tool(case: BenchmarkCase, output: str) -> float:
    try:
        parsed = json_from_output(output)
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    expected = case.expected
    if case.source_id == "arabic-function-calling":
        name = parsed.get("functionName", parsed.get("name"))
        return float(name == expected["functionName"] and parsed.get("arguments") == expected["arguments"])
    if case.source_id == "arabic-agent-eval":
        return float(parsed.get("tool_calls") == expected["tool_calls"])
    fields = ("action", "tool_name", "missing_required_arguments", "gold_call")
    return mean(float(parsed.get(field) == expected.get(field)) for field in fields)


def _ifeval(case: BenchmarkCase, output: str) -> float:
    checks: list[bool] = []
    ids = case.expected.get("instructionIds", [])
    kwargs: list[dict[str, Any]] = []
    raw = case.expected.get("kwargs")
    if isinstance(raw, str):
        try:
            kwargs = json.loads(raw)
        except json.JSONDecodeError:
            kwargs = []
    for index, instruction in enumerate(ids):
        values = kwargs[index] if index < len(kwargs) else {}
        if "number_paragraphs" in instruction or "multiple_sections" in instruction:
            expected = int(values.get("num_paragraphs", values.get("num_sections", 3)))
            separator = values.get("section_spliter", "* * *")
            checks.append(len([part for part in output.split(separator) if part.strip()]) == expected)
        elif "number_words_at_least" in instruction:
            checks.append(len(output.split()) >= int(values.get("num_words", 150)))
        elif "end" in instruction:
            phrase = values.get("end_phrase") or values.get("end")
            checks.append(bool(phrase) and output.rstrip().endswith(str(phrase)))
        else:
            checks.append(bool(output.strip()))
    return mean(checks) if checks else float(bool(output.strip()))


def _mena(case: BenchmarkCase, output: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", output)
    if not match:
        return 0.0
    selected = str(int(float(match.group())))
    by_option: dict[str, list[float]] = {}
    for raw in case.expected["surveyDistributions"].values():
        try:
            _, distribution = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        for option, percent in distribution.items():
            by_option.setdefault(str(option), []).append(float(str(percent).rstrip("%")))
    probabilities = {key: mean(values) for key, values in by_option.items()}
    maximum = max(probabilities.values(), default=0)
    return probabilities.get(selected, 0) / maximum if maximum else 0.0


def deterministic_grade(
    case: BenchmarkCase, output: str, *, fixture_result: dict | None = None
) -> dict[str, Any] | None:
    if case.fixture:
        if fixture_result is None:
            return {"method": "requires-fixture-harness", "score": None, "passed": None}
        from najd_benchmark.fixtures import artifact_grade
        return artifact_grade(case.expected, fixture_result)
    if "missing_expected_answer" in case.audit_issues:
        return {"method": "missing-reference-answer", "score": None, "passed": None}
    if case.provenance.get("correction_version") == "2026.09.27.1":
        from najd_benchmark.absher_corrections import grade
        return grade(case.expected, output)
    if case.source_id == "arabicmmlu":
        return grade_arabic_mmlu(case.to_dict(), output)
    mode = adapt(case).mode
    if mode == "judge":
        return None
    if mode == "tool":
        value = _tool(case, output)
    elif mode == "ifeval":
        value = _ifeval(case, output)
    elif mode == "mena":
        value = _mena(case, output)
    elif "answerKey" in case.expected or "answerIndex" in case.expected:
        value = _choice(case, output)
        if mcq_options(case):
            return {"method": CHOICE_METHOD, "score": round(value, 6), "passed": value == 1.0}
    else:
        value = _structured(case, output)
    return {"method": f"{mode}@1", "score": round(value, 6), "passed": value == 1.0}


def aggregate(grades: list[dict[str, Any]]) -> dict[str, Any]:
    by_track: dict[str, list[float]] = {}
    for grade in grades:
        if grade.get("score") is not None:
            by_track.setdefault(grade["track"], []).append(float(grade["score"]))
    tracks = {track: round(mean(scores), 6) for track, scores in sorted(by_track.items())}
    scores = [score for values in by_track.values() for score in values]
    return {
        "najd_score": round(mean(tracks.values()), 6) if tracks else None,
        "case_weighted_score": round(mean(scores), 6) if scores else None,
        "tracks": tracks,
        "graded_cases": len(scores),
        "coverage": len(scores) / len(grades) if grades else 0,
    }

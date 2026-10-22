from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Literal

from najd_benchmark.arabic_mmlu import build_messages

from .models import BenchmarkCase

GradeMode = Literal["exact", "tool", "ifeval", "mena", "judge"]

MCQ_KEYS = {"answerKey", "answerIndex"}
MCQ_LETTER_INSTRUCTION = "Return only the letter of the correct option, without explanation."
OPTION_LETTERS = string.ascii_uppercase
# Arabic letters for option positions 0-4, with the variants seen in stored option labels.
ARABIC_OPTION_LETTERS = ("أاإآ", "ب", "ج", "د", "ه")
TOOL_SOURCES = {"arabic-function-calling", "arabic-agent-eval", "paired-msa-saudi-tool-use"}
JUDGE_SOURCES = {
    "QCRI/IslamicFaithQA", "arbml-arabic-rc", "arabicragb", "pico-saudi-v0.01",
    "arabic-safety-evaluation",
}
IFEVAL_SOURCES = {"humain-araifeval", "inception-arabic-ifeval"}


@dataclass(frozen=True)
class AdaptedCase:
    messages: list[dict[str, str]]
    mode: GradeMode
    output_contract: str


def _instruction(case: BenchmarkCase) -> tuple[GradeMode, str]:
    expected = case.expected
    if case.provenance.get("correction_version") == "2026.09.27.1":
        return "exact", "Return only the Arabic option key (أ، ب، ج، د), without explanation."
    if case.source_id in TOOL_SOURCES:
        return "tool", (
            "Return only JSON describing the requested tool action. Include the function/tool name, "
            "arguments, and whether clarification is required. Do not execute the action."
        )
    if case.source_id in IFEVAL_SOURCES:
        return "ifeval", "Follow every instruction in the user prompt exactly."
    if case.source_id == "mena-values":
        return "mena", "Return only one numeric value from the requested scale."
    if case.source_id in JUDGE_SOURCES or set(expected) == {"answer"}:
        return "judge", "Answer the question directly and faithfully in the requested language."
    if MCQ_KEYS & set(expected):
        if mcq_options(case):
            return "exact", MCQ_LETTER_INSTRUCTION
        return "exact", "Return only the answer option key or index, without explanation."
    if set(expected) & {"label", "Label", "dangerous", "safetyLabel", "topic", "hate"}:
        labels = json.dumps(expected_contract(case), ensure_ascii=False)
        return "exact", f"Return only JSON matching this field contract: {labels}"
    return "judge", "Answer directly. When classification is requested, return concise JSON."


def adapt(case: BenchmarkCase) -> AdaptedCase:
    if case.source_id == "arabicmmlu":
        messages = build_messages(case.to_dict())
        return AdaptedCase(messages, "exact", "arabic-mmlu-key-v1")
    mode, instruction = _instruction(case)
    messages: list[dict[str, str]] = []
    system = "You are being evaluated by Najd Arena. " + instruction
    if case.system_prompt:
        system += "\n\n" + case.system_prompt
    messages.append({"role": "system", "content": system})
    user = case.prompt
    options = mcq_options(case) if instruction == MCQ_LETTER_INSTRUCTION else None
    if options:
        texts = option_texts(options)
        if not _options_listed(user, texts):
            user += "\n\n" + "\n".join(f"{OPTION_LETTERS[i]}. {text}" for i, text in enumerate(texts))
    messages.append({"role": "user", "content": user})
    return AdaptedCase(messages, mode, instruction)


def mcq_options(case: BenchmarkCase) -> list[str] | None:
    """The stored options of a multiple-choice case, or None when it has no usable options."""
    options = case.expected.get("options") if MCQ_KEYS & set(case.expected) else None
    if not isinstance(options, list) or not 0 < len(options) <= len(OPTION_LETTERS):
        return None
    return [str(option) for option in options]


def strip_option_label(option: str, position: int) -> str:
    """Remove a leading label that names this option's own position, e.g. "A. x", "(A) x", "أ) x"."""
    letters = OPTION_LETTERS[position]
    if position < len(ARABIC_OPTION_LETTERS):
        letters += ARABIC_OPTION_LETTERS[position]
    label = rf"(?:\(\s*[{letters}]\s*\)|[{letters}]\u0640?\s*[.):\-\u2013])"
    stripped = re.sub(rf"^\s*{label}\s*", "", option, count=1, flags=re.IGNORECASE)
    return stripped or option


def option_texts(options: list[str]) -> list[str]:
    return [strip_option_label(option, position) for position, option in enumerate(options)]


def _squash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _options_listed(prompt: str, texts: list[str]) -> bool:
    """True when the prompt already lists every option on its own line, so appending would repeat them."""
    lines = [_squash(line) for line in prompt.splitlines()]
    return all(
        _squash(text) and any(_squash(strip_option_label(line, position)) == _squash(text) for line in lines)
        for position, text in enumerate(texts)
    )


def expected_contract(case: BenchmarkCase) -> dict[str, Any]:
    expected = case.expected
    if "answerKey" in expected:
        return {"answerKey": "string"}
    if "answerIndex" in expected:
        return {"answerIndex": "integer"}
    return {key: type(value).__name__ for key, value in expected.items()}


def json_from_output(output: str) -> Any:
    text = output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def normalize_token(value: Any) -> str:
    translation = str.maketrans({
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "أ": "a", "ا": "a", "ب": "b", "ج": "c", "د": "d", "ه": "e",
    })
    return re.sub(r"[\s\W_]+", "", str(value).translate(translation).casefold())


def supported_sources() -> frozenset[str]:
    return frozenset({
        "QCRI/IslamicFaithQA", "absher", "alghafa-native", "arabic-agent-eval",
        "arabic-exams", "arabic-function-calling", "arabic-safety-evaluation",
        "arabicmmlu", "arabicragb", "arasafe", "aratrust", "arbml-arabic-hate-speech",
        "arbml-arabic-rc", "arbml-arabic_dialects_dataset", "arbml-cidar-eval-100",
        "arbml-cidar-mcq-100", "arbml-dangerous-dataset", "arbml-quran_hadith",
        "arbml-saudiirony", "commonsense-validation", "dialectal-arabic-mmlu",
        "humain-araifeval", "humain-aramath", "humain-arapro", "humain-aratruthfulqa",
        "inception-arabic-ifeval", "mena-values", "paired-msa-saudi-tool-use",
        "pico-saudi-v0.01", "najd-benchmark-v1", "almrsal-general-contest",
        "almrsal-riddles", "mawdoo3-animals", "mawdoo3-riddles", "mawdoo3-science",
        "qusama-riddles", "sayidaty-children-riddles", "twinkl-arabic-riddles",
        "twinkl-islamic-questions",
    })

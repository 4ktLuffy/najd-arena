from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from najd_arena import engine
from najd_arena.models import Benchmark, BenchmarkCase, JudgeConfig, ModelConfig, RunConfig
from najd_arena.providers import ModelResponse


def make_case(case_id: str, source: str, expected: dict) -> BenchmarkCase:
    return BenchmarkCase(case_id, "arabic", "ar", "question", (), expected, {"sourceId": source}, ())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run(tmp_path: Path, cases: tuple[BenchmarkCase, ...], *, judge: JudgeConfig | None = None) -> Path:
    benchmark = Benchmark("test", "version", "revision", "sha", tmp_path, cases)
    response = ModelResponse("fake", '{"functionName":"weather","arguments":{}}', [], {})
    config = RunConfig(model=ModelConfig("fake", "fake/model"), judge=judge, max_attempts=1)
    with (patch("najd_arena.engine.fetch_benchmark", return_value=benchmark),
          patch("najd_arena.engine.LiteLLMProvider.invoke", new=AsyncMock(return_value=response))):
        run_id = await engine.run_benchmark(tmp_path, config)
    return tmp_path / ".najd-arena-v1" / "runs" / run_id


@pytest.mark.asyncio
async def test_grader_exception_is_recorded_and_run_completes(tmp_path: Path) -> None:
    cases = (
        make_case("good", "arabic-function-calling", {"functionName": "weather", "arguments": {}}),
        make_case("bad", "arabic-function-calling", {"functionName": "weather", "arguments": {}}),
    )
    real_grade = engine.deterministic_grade

    def flaky_grade(case: BenchmarkCase, output: str, **kwargs):
        if case.id == "bad":
            raise RuntimeError("grader blew up")
        return real_grade(case, output, **kwargs)

    with patch("najd_arena.engine.deterministic_grade", new=flaky_grade):
        run_dir = await run(tmp_path, cases)
    grades = {row["case_id"]: row for row in read_jsonl(run_dir / "grades.jsonl")}
    assert grades["good"]["score"] == 1
    assert {key: grades["bad"][key] for key in ("method", "score", "passed", "error")} == {
        "method": "grading-error", "score": None, "passed": None, "error": "RuntimeError: grader blew up"}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["graded_cases"] == 1
    assert report["coverage"] == 0.5


@pytest.mark.asyncio
async def test_judge_exception_is_recorded_and_run_completes(tmp_path: Path) -> None:
    cases = (make_case("open", "QCRI/IslamicFaithQA", {"answer": "x"}),)
    with patch("najd_arena.engine._judge", new=AsyncMock(side_effect=ValueError("Judge returned garbage"))):
        run_dir = await run(tmp_path, cases, judge=JudgeConfig("judge/model", "JUDGE_KEY"))
    [row] = read_jsonl(run_dir / "grades.jsonl")
    assert (row["method"], row["score"], row["passed"]) == ("grading-error", None, None)
    assert row["error"] == "ValueError: Judge returned garbage"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["coverage"] == 0

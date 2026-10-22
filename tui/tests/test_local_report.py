from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from najd_arena import engine
from najd_arena.models import Benchmark, BenchmarkCase, ModelConfig, RunConfig
from najd_arena.providers import ModelResponse
from najd_arena.tui import ArenaApp


def tool_case(case_id: str, prompt: str) -> BenchmarkCase:
    return BenchmarkCase(case_id, "arabic", "ar", prompt, (), {"functionName": "weather", "arguments": {}},
                         {"sourceId": "arabic-function-calling"}, ())


async def run(tmp_path: Path, prompts: dict[str, str]) -> tuple[dict, dict]:
    """Run a fake model that answers correctly unless the prompt is 'fail' (API error)."""
    cases = tuple(tool_case(case_id, prompt) for case_id, prompt in prompts.items())
    benchmark = Benchmark("test", "version", "revision", "sha", tmp_path, cases)

    async def invoke(self, messages):
        if messages[-1]["content"] == "fail":
            raise RuntimeError("api down")
        return ModelResponse("fake", '{"functionName":"weather","arguments":{}}', [], {})

    config = RunConfig(model=ModelConfig("fake", "fake/model"), max_attempts=1)
    with (patch("najd_arena.engine.fetch_benchmark", return_value=benchmark),
          patch("najd_arena.engine.LiteLLMProvider.invoke", new=invoke)):
        run_id = await engine.run_benchmark(tmp_path, config)
    run_dir = tmp_path / ".najd-arena-v1" / "runs" / run_id
    return (json.loads((run_dir / "report.json").read_text(encoding="utf-8")),
            json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")))


@pytest.mark.asyncio
async def test_partial_run_reports_no_headline_score(tmp_path: Path) -> None:
    report, manifest = await run(tmp_path, {"a": "ok", "b": "fail", "c": "fail"})
    assert report["najd_score"] is None
    assert report["case_weighted_score"] is None
    assert report["status"] == "incomplete"
    assert report["coverage"] == pytest.approx(1 / 3)
    assert report["errors"] == 2
    assert report["graded_cases"] == 1
    assert manifest["status"] == "incomplete"


@pytest.mark.asyncio
async def test_complete_run_keeps_headline_score(tmp_path: Path) -> None:
    report, manifest = await run(tmp_path, {"a": "ok", "b": "ok"})
    assert report["najd_score"] == 1.0
    assert report["case_weighted_score"] == 1.0
    assert (report["status"], report["coverage"], report["errors"]) == ("complete", 1.0, 0)
    assert manifest["status"] == "complete"


@pytest.mark.asyncio
async def test_run_table_shows_dash_for_missing_score() -> None:
    runs = [
        {"run_id": "partial", "target": {"model": "m"}, "status": "incomplete",
         "report": {"najd_score": None, "case_weighted_score": None, "coverage": 0.25}},
        {"run_id": "full", "target": {"model": "m"}, "status": "complete",
         "report": {"najd_score": 0.5, "coverage": 1.0}},
        {"run_id": "running", "target": {"model": "m"}, "status": "running"},
    ]
    with patch("najd_arena.tui.list_runs", return_value=runs):
        async with ArenaApp().run_test() as pilot:
            table = pilot.app.query_one("#runs")
            assert [table.get_row_at(index) for index in range(3)] == [
                ["partial", "m", "incomplete", "—", "25%"],
                ["full", "m", "complete", "0.5", "100%"],
                ["running", "m", "running", "—", "0%"],
            ]

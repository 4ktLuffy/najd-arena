from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from najd_benchmark.release import CASE_COUNT, DATASET_REVISION, DATASET_VERSION

from najd_arena.metrics import deterministic_grade
from najd_arena.models import Benchmark, BenchmarkCase
from najd_arena.worker import _benchmark_for_run, summarize_run


def benchmark():
    case = BenchmarkCase.from_dict({
        'id': 'one', 'track': 'arabic', 'language': 'ar', 'prompt': 'question',
        'expected': {'answer': 'answer'}, 'provenance': {'sourceId': 'najd-benchmark-v1'},
    })
    return Benchmark('test', 'version', 'revision', 'sha', Path('.'),
                     (case, replace(case, id='two')))


def test_run_uses_recorded_release():
    with patch('najd_arena.worker.fetch_benchmark', return_value=benchmark()) as fetch:
        _benchmark_for_run({'dataset_version': 'old', 'dataset_revision': 'immutable', 'total_cases': 2})
        fetch.assert_called_once_with(version='old', revision='immutable')
        with pytest.raises(ValueError):
            _benchmark_for_run({'dataset_version': 'old', 'dataset_revision': 'immutable', 'total_cases': 3})


def test_incomplete_results_have_no_headline():
    _, coverage, macro, micro, complete = summarize_run(benchmark(), [('one', 'arabic', 1), ('two', 'arabic', None)])
    assert (coverage, macro, micro, complete) == (0.5, None, None, False)
    assert summarize_run(benchmark(), [('one', 'arabic', 1), ('two', 'arabic', 0)])[1:] == (1, 0.5, 0.5, True)


@pytest.mark.parametrize('rows', [
    [('one', 'arabic', 1), ('one', 'arabic', 1)], [('unknown', 'arabic', 1)],
    [('one', 'wrong-track', 1)], [('one', 'arabic', float('nan'))],
])
def test_invalid_grades_rejected(rows):
    with pytest.raises(ValueError):
        summarize_run(benchmark(), rows)


def test_missing_evidence_stays_ungraded():
    case = benchmark().cases[0]
    assert deterministic_grade(replace(case, fixture='documents'), 'answer')['score'] is None
    assert deterministic_grade(replace(case, audit_issues=('missing_expected_answer',)), 'answer')['score'] is None


def test_web_and_worker_release_agree():
    source = (Path(__file__).parents[2] / 'web/lib/dataset-release.ts').read_text()
    assert DATASET_VERSION in source
    assert DATASET_REVISION in source
    assert str(CASE_COUNT) in source
    from najd_arena.engine import PROTOCOL
    assert f'protocol: "{PROTOCOL}"' in source


def test_fixture_artifacts_must_exist():
    case = replace(benchmark().cases[0], fixture='contacts',
                   expected={'file_contains': {'answer.txt': 'Lulwa'}})
    from najd_benchmark.fixtures import PROTOCOL
    evidence = {'protocol': PROTOCOL, 'status': 'complete', 'files': {}}
    assert deterministic_grade(case, 'Lulwa', fixture_result=evidence)['score'] == 0
    evidence['files']['answer.txt'] = 'Lulwa'
    assert deterministic_grade(case, 'Lulwa', fixture_result=evidence) is None


def test_corrected_keys_are_not_judged_or_guessed():
    from najd_arena.adapters import adapt
    case = replace(benchmark().cases[0], expected={'answer': 'ب'},
                   provenance={'sourceId': 'absher', 'correction_version': '2026.09.27.1'})
    assert adapt(case).mode == 'exact'
    assert deterministic_grade(case, 'ب')['score'] == 1
    assert deterministic_grade(case, 'أ')['score'] == 0
    assert deterministic_grade(case, 'أ أو ب')['score'] == 0


@pytest.mark.asyncio
async def test_invalid_judge_response_is_an_error():
    from unittest.mock import AsyncMock

    from najd_arena.engine import _judge
    from najd_arena.models import ModelConfig
    from najd_arena.providers import ModelResponse
    response = ModelResponse('judge', '{"score":2}', [], {})
    with (patch('najd_arena.engine.LiteLLMProvider.invoke', new=AsyncMock(return_value=response)),
          pytest.raises(ValueError, match='invalid structured verdict')):
        await _judge(benchmark().cases[0], 'candidate', ModelConfig('judge', 'test'))

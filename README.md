# Najd Arena

[Managed evaluation plan](docs/managed-evaluations.md) · [Contributing](CONTRIBUTING.md)

Najd Arena is the public results and managed evaluation application for the [Najd Benchmark](https://huggingface.co/datasets/najdresearch/najd-benchmark). It contains two products:

- `tui/` — a local Textual application, scriptable CLI, evaluation runtime, and reusable worker process.
- `web/` — a Next.js leaderboard and authenticated organization workspace.

Shared JSON contracts live in `contracts/`; portable deployment files live in `infra/`. There is no separate application backend: the Next.js app owns HTTP/auth flows and the Python runtime owns queued evaluation work.

## Local TUI

```bash
cd tui
uv sync --extra dev
uv run najd-arena
```

Headless OpenAI-compatible example:

```bash
export LOCAL_MODEL_API_KEY=unused
uv run najd-arena run \
  --model openai/local-model \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_MODEL_API_KEY \
  --sample 20
```

Use a LiteLLM model string and its standard environment variable for hosted providers. Add `--judge-model` and `--judge-api-key-env` for full hybrid grading. Without a judge, open-ended cases remain ungraded and the report is non-canonical.

## Web and workers

```bash
cp .env.example .env
# Fill OAuth, credential-encryption, admin, and judge values.
docker compose -f infra/docker-compose.yml up --build
```

Generate `ARENA_CREDENTIAL_KEY` as 32 random bytes encoded with URL-safe base64. Target credentials are encrypted for one run and erased after inference. Hosted endpoints must be public HTTPS URLs.

The first upstream organization syncs on GitHub or Hugging Face sign-in. Organizations begin unapproved with zero quota; approve them and set `active_run_limit` and `monthly_run_limit` in PostgreSQL before they can launch runs.

## Scoring

The runtime pins the unified `2026.09.27.1` release: 6,089 cases across 25 tracks. It verifies the case checksum and preserves each run's dataset version/revision. ArabicMMLU scoring comes from the pinned benchmark package. Structured tasks use deterministic metrics; open-ended tasks require a configured judge.

Multiple-choice cases other than ArabicMMLU show every stored option to the model as `A.`, `B.`, … in stored order (a label that already names the option's own position is removed first) and ask for the letter only. Grading derives the correct position from the case (a Latin or Arabic letter key, a 0-based digit-string key, or a 1-based `answerIndex`) and accepts that letter or the exact text of that option; other letters and partial text score zero. These runs carry protocol `najd-prompt-v3` and `exact-choice@2` grades. Multiple-choice scores from `najd-prompt-v2-fixtures` runs did not show these options and are not comparable.

Only complete canonical runs enter the review queue. Organization-submitted publication requires organization-admin consent and at least one Najd-admin approval. Raw model outputs remain private to the organization and Najd reviewers.

## Historical M3 comparison

The separate `/historical/m3` page reads an archived HUMAIN M3 versus MiniMax M3 experiment from PostgreSQL. Its headline includes all 6,089 case IDs in each of 28 configurations, for 170,492 grade records. Correct and possible-correct grades count as acceptable; technical failures stay in the denominator. This is a historical pre-audit study, not a canonical leaderboard run. The archived prompts and expected answers differ from the published Hugging Face release on some case IDs, and the page discloses the counts.

To import it, apply the database migrations and run `tui/scripts/seed_historical_m3.py` with `--snapshot` pointing to the private archived `preaudit-snapshot-20260909T214500Z` directory and `--release` pointing to the locally verified `2026.09.14` Hugging Face release directory. Pass `--write` and `DATABASE_URL` to seed PostgreSQL. The importer checks both release-file hashes, the snapshot hashes, all case IDs, and complete configuration coverage before writing. It stores only grades and audit metadata, not model answers. Run without `--write` to inspect the verification summary.

The runner loads all 6,089 cases. The six fixture tasks use `fixture-tools-v1`, a bounded virtual filesystem with recorded reads, writes, output artifacts and grounded judging. Four formerly missing answers use documented source-based corrections in release `2026.09.27.1`. These six tasks measure harness-assisted performance, not raw single-turn inference. Infrastructure failures and invalid judge responses remain errors.

## Development

```bash
cd tui && uv run ruff check . && uv run python -m pytest -q
cd ../web && pnpm lint && pnpm build
```

## Current data and repository direction

| Component | Responsibility / current state |
|---|---|
| [datasets](https://github.com/najdresearch/datasets) | Collection, cleaning, generation and Hugging Face publication |
| [benchmark](https://github.com/najdresearch/benchmark) | Shared task execution/scoring destination; ArabicMMLU scorer is implemented |
| Arena | Public result catalog, organization UI, API and worker orchestration; general execution still lives in `tui/` pending extraction |

Current Hugging Face datasets are [Najd Benchmark](https://huggingface.co/datasets/najdresearch/najd-benchmark), [Najd Legacy 31](https://huggingface.co/datasets/najdresearch/najd-legacy-31) and [Arabic Riddles and Questions](https://huggingface.co/datasets/najdresearch/arabic-riddles). The latter two overlap the historical benchmark. Their current versions omit review annotations and retain source attribution.

New runs pin dataset revision `3fa471f6c8ed2ebc37a8b60d40c88170d682b569`. The benchmark package is pinned in `tui/pyproject.toml` and `tui/uv.lock`. Existing runs use their recorded version/revision, and historical import scripts retain their original hashes. Public model results are not regraded by this migration.

## Managed evaluation roadmap

The [managed evaluation plan](docs/managed-evaluations.md) distinguishes implementation targets from current behavior:

- `najdarena.com`: public results and methodology.
- `platform.najdarena.com`: planned organization portal for approved task packs and private reports.
- `admin.najdarena.com`: planned operations and publication administration.
- Approved organizations will self-launch approved packs within quotas; production execution remains gated on infrastructure and tenant-isolation checks.
- CLI/imported results are ineligible for new audited Arena publication. Existing historical imports remain clearly historical.
- Organization runs require both organization and Najd approval. Independent Najd evaluations will follow a separately labeled Najd-admin approval path.
- Approvals must be bound to an immutable result revision; that versioned workflow and worker hardening remain implementation work.

The first private pilot is Najd's Arabic customer-support version comparison. Initial Evaluation-as-a-Service provides private evaluation and failure analysis. Hands-on improvements to client datasets, prompts, harnesses and models are a separate service.

See [database result catalog](docs/database-result-catalog.md), [deployment runbook](docs/server-deployment.md) and [contributing](CONTRIBUTING.md).

## Shared contracts (Step 2)

Najd admins can use `/admin/contract-preview` to inspect private development bundles
from the benchmark package. This validates the shared format without importing local
results into Arena's publication workflow. See [integration and contribution checks](docs/shared-contracts.md).

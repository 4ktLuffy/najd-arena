from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ProgressBar, Static

from .engine import list_runs, run_benchmark
from .models import JudgeConfig, ModelConfig, RunConfig


class ArenaApp(App[None]):
    CSS = """
    Screen { background: #0d100e; color: #f7f7f2; }
    #setup { width: 38; padding: 1 2; border-right: solid #42b978; }
    #main { padding: 1 2; }
    Input { margin-bottom: 1; }
    Button { width: 100%; background: #006c35; margin-top: 1; }
    DataTable { height: 1fr; margin-top: 1; }
    #status { color: #42b978; margin: 1 0; }
    """
    TITLE = "Najd Arena"
    SUB_TITLE = "Arabic-first model evaluation"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="setup"):
                yield Label("MODEL")
                yield Input(placeholder="openai/gpt-4.1-mini or ollama/qwen3", id="model")
                yield Label("OPENAI-COMPATIBLE BASE URL")
                yield Input(placeholder="http://127.0.0.1:11434/v1", id="base")
                yield Label("API KEY ENVIRONMENT VARIABLE")
                yield Input(placeholder="OPENAI_API_KEY", id="key-env")
                yield Label("SAMPLE (blank = full suite)")
                yield Input(placeholder="20", id="sample", type="integer")
                yield Label("JUDGE MODEL (optional)")
                yield Input(placeholder="openai/judge-model", id="judge-model")
                yield Label("JUDGE KEY ENVIRONMENT VARIABLE")
                yield Input(placeholder="OPENAI_API_KEY", id="judge-key-env")
                yield Button("Start benchmark", id="run", variant="success")
            with Vertical(id="main"):
                yield Static("Ready. A full run contains 6,089 cases.", id="status")
                yield ProgressBar(total=100, show_eta=True, id="progress")
                yield DataTable(id="runs", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.add_columns("Run", "Model", "Status", "Najd score", "Coverage")
        self.refresh_runs()

    def refresh_runs(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear()
        for run in list_runs(Path.cwd()):
            report = run.get("report", {})
            score = report.get("najd_score")
            table.add_row(run["run_id"], run["target"]["model"], run["status"],
                          "—" if score is None else str(score),
                          f"{report.get('coverage', 0):.0%}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            return
        model = self.query_one("#model", Input).value.strip()
        if not model:
            self.query_one("#status", Static).update("Enter a LiteLLM model string first.")
            return
        sample_value = self.query_one("#sample", Input).value.strip()
        judge_model = self.query_one("#judge-model", Input).value.strip()
        judge_key_env = self.query_one("#judge-key-env", Input).value.strip()
        if judge_model and not judge_key_env:
            self.query_one("#status", Static).update("Enter the judge key environment variable.")
            return
        config = RunConfig(model=ModelConfig(
            name=model, model=model,
            api_base=self.query_one("#base", Input).value.strip() or None,
            api_key_env=self.query_one("#key-env", Input).value.strip() or None,
        ), judge=JudgeConfig(judge_model, judge_key_env) if judge_model else None,
            sample=int(sample_value) if sample_value else None)
        event.button.disabled = True
        self.execute(config)

    @work(exclusive=True)
    async def execute(self, config: RunConfig) -> None:
        status = self.query_one("#status", Static)
        progress = self.query_one("#progress", ProgressBar)

        def update(done: int, total: int, case_id: str) -> None:
            progress.update(total=total, progress=done)
            status.update(f"{done:,}/{total:,} · {case_id}")

        try:
            run_id = await run_benchmark(Path.cwd(), config, progress=update)
            status.update(f"Complete · {run_id}")
        except Exception as exc:
            status.update(f"Run failed: {exc}")
        finally:
            self.query_one("#run", Button).disabled = False
            self.refresh_runs()


def launch() -> None:
    ArenaApp().run()

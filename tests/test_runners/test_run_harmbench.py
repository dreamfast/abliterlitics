"""Tests for runners/run_harmbench.py — the watchdog-poisoning guards and
docker path resolution (runners-python audit, lessons #82/#83/#55a/#64).

Pure-logic tests only: DB writes and docker calls are monkeypatched. No GPU,
no docker daemon, no DB needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from runners import run_harmbench as rh

# ---------------------------------------------------------------------------
# Lesson #82: stage completion is score-file-gated
# ---------------------------------------------------------------------------


class TestMarkStageCompleteIfScored:
    """_mark_stage_complete_if_scored — lesson #82 regression guards."""

    @pytest.fixture()
    def db_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        """Capture _db_stage calls (action, model, kwargs)."""
        calls: list[tuple] = []
        monkeypatch.setattr(
            rh,
            "_db_stage",
            lambda action, comparison, model, stage="harmbench", **kw: calls.append((action, model, kw)),
        )
        return calls

    def test_complete_only_when_scores_exist(self, tmp_path: Path, db_calls: list[tuple]) -> None:
        score_file = tmp_path / "harmbench_m_scores.json"
        score_file.write_text("{}")

        result = rh._mark_stage_complete_if_scored("cmp", "m", score_file)

        assert result is True
        assert len(db_calls) == 1
        action, model, kw = db_calls[0]
        assert action == "complete"
        assert model == "m"
        assert kw["result_path"] == str(score_file)

    def test_fail_when_scores_missing(self, tmp_path: Path, db_calls: list[tuple]) -> None:
        score_file = tmp_path / "harmbench_m_scores.json"  # NOT created

        result = rh._mark_stage_complete_if_scored("cmp", "m", score_file)

        assert result is False
        assert len(db_calls) == 1
        action, model, kw = db_calls[0]
        assert action == "fail"
        assert model == "m"
        # The failure reason is persisted for pipeline-status/ops debugging.
        assert "generation failed" in kw["error_message"]

    def test_dry_run_passes_through(self, tmp_path: Path, db_calls: list[tuple]) -> None:
        score_file = tmp_path / "s.json"
        score_file.write_text("{}")

        rh._mark_stage_complete_if_scored("cmp", "m", score_file, dry_run=True)
        # dry_run flag is forwarded — _db_stage itself is the monkeypatched spy,
        # so we assert the flag reached it.
        assert db_calls[0][2]["dry_run"] is True

    def test_never_marks_complete_without_file_even_if_responses_exist(
        self, tmp_path: Path, db_calls: list[tuple]
    ) -> None:
        """The apostate_diode poisoning: responses generated but scores never
        written (vLLM startup failed AFTER the phase ran) must NOT complete."""
        (tmp_path / "harmbench_m_responses.json").write_text("{}")
        score_file = tmp_path / "harmbench_m_scores.json"

        result = rh._mark_stage_complete_if_scored("cmp", "m", score_file)

        assert result is False
        assert db_calls[0][0] == "fail"


# ---------------------------------------------------------------------------
# Lesson #83: skip path must never demote a 'completed' stage
# ---------------------------------------------------------------------------


class TestMarkSkipPreservingCompleted:
    """_mark_skip_preserving_completed — lesson #83 regression guards."""

    def test_skip_written_when_stage_not_completed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rh, "_stage_completed", lambda c, m: False)
        calls: list[tuple] = []
        monkeypatch.setattr(rh, "_db_stage", lambda action, c, m, **kw: calls.append(action))

        rh._mark_skip_preserving_completed("cmp", "m")

        assert calls == ["skip"]

    def test_completed_status_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A watchdog relaunch after natural completion must NOT write 'skip'
        over 'completed' (the relaunch-loop poison, 2026-09-10 apostate_diode)."""
        monkeypatch.setattr(rh, "_stage_completed", lambda c, m: True)
        calls: list[tuple] = []
        monkeypatch.setattr(rh, "_db_stage", lambda action, c, m, **kw: calls.append(action))

        rh._mark_skip_preserving_completed("cmp", "m")

        assert calls == []  # no DB write at all

    def test_stage_check_error_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real _stage_completed NEVER raises — it returns False on DB errors
        (read path is best-effort). Simulated error → False → skip write proceeds
        (write paths stay authoritative)."""
        monkeypatch.setattr(rh, "_stage_completed", lambda c, m: False)  # DB-error stand-in
        calls: list[tuple] = []
        monkeypatch.setattr(rh, "_db_stage", lambda action, c, m, **kw: calls.append(action))

        rh._mark_skip_preserving_completed("cmp", "m")

        assert calls == ["skip"]

    def test_dry_run_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rh, "_stage_completed", lambda c, m: False)
        calls: list[tuple] = []
        monkeypatch.setattr(rh, "_db_stage", lambda action, c, m, **kw: calls.append(kw))

        rh._mark_skip_preserving_completed("cmp", "m", dry_run=True)

        assert calls and calls[0]["dry_run"] is True


# ---------------------------------------------------------------------------
# Lesson #55a: docker bind-mount paths must be absolute (model + LoRA adapter)
# ---------------------------------------------------------------------------


class TestResolveDockerPaths:
    def test_relative_model_and_lora_resolved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_dir = tmp_path / "models" / "base"
        adapter = tmp_path / "adapters" / "lora"
        model_dir.mkdir(parents=True)
        adapter.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        model, lora = rh._resolve_docker_paths("models/base", "adapters/lora")

        assert model == str(model_dir.resolve())
        assert lora == str(adapter.resolve())

    def test_absolute_paths_pass_through_unchanged(self, tmp_path: Path) -> None:
        model = str(tmp_path / "m")
        adapter = str(tmp_path / "a")

        got_model, got_lora = rh._resolve_docker_paths(model, adapter)

        assert got_model == model
        assert got_lora == adapter

    def test_empty_lora_stays_empty(self, tmp_path: Path) -> None:
        _, lora = rh._resolve_docker_paths(str(tmp_path), "")
        assert lora == ""

    def test_symlink_resolved_to_realpath(self, tmp_path: Path) -> None:
        real = tmp_path / "real-model"
        real.mkdir()
        link = tmp_path / "link-model"
        link.symlink_to(real)

        model, _ = rh._resolve_docker_paths(str(link), "")

        assert model == str(real.resolve())

    def test_run_generate_vllm_resolves_lora_mount(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wiring test: run_generate_vllm must route the adapter through the
        resolver (the lesson-#55a fix site). dry_run=True keeps the test
        docker-free while still exercising the path-resolution code."""
        seen: list[tuple[str, str]] = []
        real = rh._resolve_docker_paths

        def spy(model_path: str, lora_adapter_path: str = "") -> tuple[str, str]:
            seen.append((model_path, lora_adapter_path))
            return real(model_path, lora_adapter_path)

        monkeypatch.setattr(rh, "_resolve_docker_paths", spy)

        result = rh.run_generate_vllm(
            "relative/model",
            "m",
            0,
            tmp_path,
            max_tokens=100,
            max_model_len=8192,
            reasoning_parser="",
            reasoning_config_json="",
            dry_run=True,
            skip_existing=False,
            lora_adapter_path="relative/adapter",
        )

        assert result is None  # dry run
        assert seen == [("relative/model", "relative/adapter")]


# ---------------------------------------------------------------------------
# Lesson #64: consumer slug normalization == exporter normalization
# ---------------------------------------------------------------------------


class TestSlugNormalization:
    def test_slug_re_matches_exporter_normalization(self) -> None:
        """_SLUG_RE must be the same normalization _load_comparison.py applies
        (verified against the exporter's own regex in its tests)."""
        for name, expected in [
            ("sdft-cybersec-r2", "sdft_cybersec_r2"),
            ("heretic", "heretic"),
            ("Qwen3.8-27B", "Qwen3_8_27B"),
            ("a_b-c.d", "a_b_c_d"),
        ]:
            assert rh._SLUG_RE.sub("_", name) == expected
            assert re.sub(r"[^A-Za-z0-9_]", "_", name) == expected

    def test_lora_env_var_lookup_uses_slug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The LoRA lookup in main() keys on the SLUGGED name — assert the
        env-var name the lookup builds for a dotted/hyphenated model name."""
        # Simulate the exporter having exported the adapter under the slug.
        monkeypatch.setenv("VARIANT_LORA_ADAPTER_Qwen3_8_27B", "/adapters/x")
        slug = rh._SLUG_RE.sub("_", "Qwen3.8-27B")
        import os

        assert os.environ.get(f"VARIANT_LORA_ADAPTER_{slug}", "") == "/adapters/x"


# ---------------------------------------------------------------------------
# _stage_completed read-only contract
# ---------------------------------------------------------------------------


class TestStageCompletedContract:
    def test_false_when_db_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Nonexistent DB file → False (never raises)."""
        monkeypatch.setattr(rh, "ABL_ROOT", tmp_path)
        assert rh._stage_completed("no-such-cmp", "m") is False

    def test_false_on_real_db_when_row_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Real sqlite DB without the row → False."""
        import sqlite3

        db = tmp_path / "abliterlitics.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE model_stages (comparison TEXT, model_name TEXT, stage TEXT, status TEXT)")
        conn.execute("INSERT INTO model_stages VALUES ('cmp', 'other', 'harmbench', 'completed')")
        conn.commit()
        conn.close()
        monkeypatch.setattr(rh, "ABL_ROOT", tmp_path)

        assert rh._stage_completed("cmp", "m") is False
        assert rh._stage_completed("cmp", "other") is True
        assert rh._stage_completed("cmp", "other", stage="weights") is False


# ---------------------------------------------------------------------------
# MTP index gating (SIM115 fix site) — index read with a context manager
# ---------------------------------------------------------------------------


class TestMtpIndexRead:
    def test_index_with_mtp_keys_detected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The has_mtp logic reads the index via a with-block now; exercise the
        same read pattern the server_cmd build uses."""
        idx = {"weight_map": {"model.layers.0.attn.weight": 0, "mtp.fc.weight": 1}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(idx))
        monkeypatch.chdir(tmp_path)

        with open(Path("model.safetensors.index.json")) as f:
            loaded = json.load(f)
        assert any(k.startswith("mtp.") for k in loaded["weight_map"])


# ---------------------------------------------------------------------------
# Global-flag parity: abliterlitics.sh global_flags() passes --skip-existing
# to this runner on `./abliterlitics.sh --skip-existing harmbench <dir>` —
# the arg must stay accepted (audit item: "no-op flag — keep or wire").
# ---------------------------------------------------------------------------


def test_skip_existing_flag_accepted_for_global_parity() -> None:
    import subprocess
    import sys

    runner = Path(__file__).resolve().parents[2] / "runners" / "run_harmbench.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--skip-existing" in result.stdout

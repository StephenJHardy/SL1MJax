from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "scripts" / "run_3c391_phase6_after_commissioning.sh"
WAITER = ROOT / "scripts" / "wait_3c391_phase6_live_then_continue.sh"
WATCHER = ROOT / "scripts" / "watch_3c391_c1_review.sh"
SYNC = ROOT / "scripts" / "sync_3c391_phase6_live_products.sh"
LADDER = ROOT / "scripts" / "run_3c391_phase6_explicit_ladder.sh"


def test_continuation_scripts_do_not_rsync_live_products() -> None:
    after = AFTER.read_text(encoding="utf-8")
    waiter = WAITER.read_text(encoding="utf-8")
    ladder = LADDER.read_text(encoding="utf-8")
    sync = SYNC.read_text(encoding="utf-8")
    for text in (after, waiter, ladder, sync):
        assert "copy_phase6_products" in text
        assert 'rsync -a "$LIVE_OUT/commissioning/' not in text
        assert 'rsync -a "$LIVE_OUT/commissioning-c4/' not in text
        assert 'rsync -a "$LIVE_OUT/source.json"' not in text


def test_continuation_scripts_use_staged_source_manifest() -> None:
    after = AFTER.read_text(encoding="utf-8")
    waiter = WAITER.read_text(encoding="utf-8")
    ladder = LADDER.read_text(encoding="utf-8")
    for text in (after, waiter, ladder):
        assert "write_staged_source_manifest" in text
        assert "preserve_commissioning_source" in text
        assert "explicit_source.json" in text
        assert '--source-manifest "$LIVE_OUT/source.json"' not in text
        assert '--source-manifest "$SOURCE_JSON"' in text


def test_waiter_requires_stable_idle_before_resume() -> None:
    waiter = WAITER.read_text(encoding="utf-8")
    assert "_live_wrapper" in waiter
    assert "_live_idle" in waiter
    assert "live job still holds the GPU; refuse" in waiter
    assert "IDLE_RECHECK" in waiter


def test_compare_wrapper_accepts_pointing_argument() -> None:
    compare = (ROOT / "scripts" / "run_3c391_operator_compare_bacchus.sh").read_text(
        encoding="utf-8"
    )
    assert 'POINTING="${3:-C1}"' in compare
    assert "--pointing \"$POINTING\"" in compare


def test_c1_watcher_fails_when_producer_exits() -> None:
    watcher = WATCHER.read_text(encoding="utf-8")
    assert "_producer_gone" in watcher
    assert "C1 producer exited without" in watcher
    assert "exit 2" in watcher

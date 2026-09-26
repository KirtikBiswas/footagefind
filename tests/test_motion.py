import numpy as np
import pytest

from footagefind.motion import MotionGate


def scene(rng, h=240, w=320):
    return rng.integers(60, 200, (h, w, 3), dtype=np.uint8)


def with_box(img, x, y, size=40, value=255):
    out = img.copy()
    out[y : y + size, x : x + size] = value
    return out


@pytest.mark.parametrize("method", ["diff", "mog2"])
def test_static_frames_are_skipped_and_moving_object_kept(rng, method):
    bg = scene(rng)
    gate = MotionGate(method=method, heartbeat_seconds=1e9, min_changed_fraction=0.005)
    kept = [gate.keep(bg, t * 0.5) for t in range(10)]          # 10 identical frames
    assert kept[0] is True                                       # first frame always kept
    assert sum(kept[1:]) == 0
    moved = [gate.keep(with_box(bg, 20 + 30 * i, 100), 5 + i * 0.5) for i in range(5)]
    assert all(moved)
    s = gate.stats.as_dict()
    assert s["frames_seen"] == 15 and s["frames_kept"] == 6 and s["frames_skipped"] == 9


def test_sensor_noise_does_not_count_as_motion(rng):
    bg = scene(rng).astype(np.float32)
    gate = MotionGate(method="diff", heartbeat_seconds=1e9)
    for i in range(20):
        noisy = np.clip(bg + rng.normal(0, 2.0, bg.shape), 0, 255).astype(np.uint8)
        gate.keep(noisy, i * 0.5)
    assert gate.stats.kept == 1


def test_heartbeat_keeps_a_frame_in_static_scene(rng):
    bg = scene(rng)
    gate = MotionGate(method="diff", heartbeat_seconds=10.0)
    kept_t = [t / 2 for t in range(0, 61) if gate.keep(bg, t / 2)]  # 30 s of a static scene
    assert kept_t == [0.0, 10.0, 20.0, 30.0]
    assert gate.stats.kept_by_heartbeat == 3


def test_disabled_gate_keeps_everything(rng):
    bg = scene(rng)
    gate = MotionGate(enabled=False)
    assert all(gate.keep(bg, t) for t in range(5))
    assert gate.stats.skipped == 0


def test_slow_drift_accumulates_against_last_kept_frame(rng):
    """Compare to the last KEPT frame, so a slowly moving object is eventually caught."""
    bg = scene(rng)
    gate = MotionGate(method="diff", heartbeat_seconds=1e9, min_changed_fraction=0.02)
    kept = [gate.keep(with_box(bg, 10 + 2 * i, 100, size=60), i * 0.5) for i in range(40)]
    assert 1 < sum(kept) < 40


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        MotionGate(method="optical-flow")

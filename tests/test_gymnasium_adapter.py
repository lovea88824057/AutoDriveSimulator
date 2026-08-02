from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldsimflow import gymnasium_available, make_gymnasium_env


def test_optional_gymnasium_adapter_imports_without_hard_dependency():
    if not gymnasium_available():
        try:
            make_gymnasium_env(ROOT / "data" / "sample_scenario.json", max_steps=2)
        except ImportError as exc:
            assert "pip install gymnasium numpy" in str(exc)
        else:
            raise AssertionError("expected ImportError when gymnasium is unavailable")
        return

    env = make_gymnasium_env(ROOT / "data" / "sample_scenario.json", max_steps=2, observation_mode="dict")
    obs, info = env.reset(seed=5)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert set(obs) == {"state", "bev"}
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "done_reason" in step_info
    env.close()

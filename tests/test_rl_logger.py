from rl.rl_logger import EpisodeLogger
import os

def test_episode_logger_roundtrip(tmp_path):
    log_path = tmp_path / "episodes.jsonl"
    logger = EpisodeLogger(str(log_path))
    logger.log(0, {"reward": 0.5, "accuracy": 0.8})
    logger.log(1, {"reward": 0.6, "accuracy": 0.82, "cost": 0.1})

    rows = logger.read_all()
    assert len(rows) == 2
    assert rows[0]["episode"] == 0
    assert "ts" in rows[0]
    assert rows[1]["reward"] == 0.6
    # path exists and non-empty
    assert os.path.getsize(log_path) > 0

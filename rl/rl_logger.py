import json
import os
import time
from typing import Any, Dict

class EpisodeLogger:
    """Simple JSONL logger for RL episodes.

    Each call to log(episode, data) appends one JSON object with mandatory fields:
      episode (int), timestamp (float)
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Touch file
        if not os.path.exists(self.path):
            with open(self.path, 'w', encoding='utf-8') as f:
                f.write("")

    def log(self, episode: int, data: Dict[str, Any]):
        record = {"episode": episode, "ts": time.time()}
        record.update(data)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

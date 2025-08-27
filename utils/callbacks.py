
from stable_baselines3.common.callbacks import BaseCallback
import csv
import os

class EpisodicCSVLogger(BaseCallback):
    """
    Logs episodic metrics (episode reward, length) to a CSV file.
    Works with Monitor wrapper in Stable-Baselines3.
    """
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._header_written = False
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Called every step; write when an episode ends
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []
        for i, done in enumerate(dones):
            if done:
                info = infos[i] if i < len(infos) else {}
                ep = info.get("episode", {})
                ep_rew, ep_len = ep.get("r"), ep.get("l")
                if ep_rew is not None and ep_len is not None:
                    self._write_row(ep_rew, ep_len, self.num_timesteps)
        return True

    def _write_row(self, ep_reward: float, ep_len: int, timesteps: int):
        write_header = not self._header_written or not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timesteps", "episode_reward", "episode_length"])
                self._header_written = True
            w.writerow([timesteps, ep_reward, ep_len])

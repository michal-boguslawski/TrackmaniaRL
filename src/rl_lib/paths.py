from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOGS_ROOT = Path("./logs")


@dataclass
class RunPaths:
    logs_root: Path
    session_id: str
    experiment: str | None
    algorithm: str | None

    @property
    def run_dir(self) -> Path:
        return self.logs_root / self.experiment / self.algorithm / self.session_id

    @property
    def log_file(self) -> Path:
        return self.run_dir / "app.log"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def video_dir(self) -> Path:
        return self.artifacts_dir / "videos"

from dataclasses import dataclass
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def worker_python(environment, directory):
    executable = ROOT / directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return os.environ.get(environment, str(executable) if executable.is_file() else sys.executable)


@dataclass(frozen=True)
class Settings:
    root: Path
    astrbot: Path = ROOT.parent / "astrbot-chat"
    rift: Path = ROOT / "vendor/rift-svc"
    rift_python: str = sys.executable
    separator_python: str = sys.executable
    music_python: str = sys.executable

    @classmethod
    def from_environment(cls):
        return cls(
            root=Path(os.environ.get("TIMBREFORGE_DATA", str(ROOT / "data"))).resolve(),
            astrbot=Path(os.environ.get("TIMBREFORGE_ASTRBOT", str(ROOT.parent / "astrbot-chat"))).resolve(),
            rift=Path(os.environ.get("TIMBREFORGE_RIFT", str(ROOT / "vendor/rift-svc"))).resolve(),
            rift_python=worker_python("TIMBREFORGE_RIFT_PYTHON", ".venv-rift"),
            separator_python=worker_python("TIMBREFORGE_SEPARATOR_PYTHON", ".venv-separator"),
            music_python=os.environ.get("TIMBREFORGE_MUSIC_PYTHON", sys.executable),
        )

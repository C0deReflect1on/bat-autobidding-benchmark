from pathlib import Path

def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find a repo marker.
    You can choose: .git, pyproject.toml, or data folder.
    """
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Repo root not found.")

# Compute once
BASE_DIR = find_repo_root(Path(__file__).resolve())
DATA_DIR = BASE_DIR / "data"

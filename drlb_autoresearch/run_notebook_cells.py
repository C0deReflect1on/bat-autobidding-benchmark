from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run_notebook(path: Path) -> None:
    payload = json.loads(path.read_text())
    cells = payload.get("cells", [])
    shared_globals: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(path),
    }
    t0 = time.time()
    print(f"[runner] start notebook={path}")
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"[runner] exec cell={idx}")
        try:
            exec(compile(source, f"{path}::cell_{idx}", "exec"), shared_globals, shared_globals)
        except Exception:
            print(f"[runner] failed cell={idx}")
            raise
    elapsed = time.time() - t0
    print(f"[runner] done notebook={path} elapsed_sec={elapsed:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Jupyter notebook code cells without nbconvert.")
    parser.add_argument("notebook", type=str)
    args = parser.parse_args()
    run_notebook(Path(args.notebook).resolve())


if __name__ == "__main__":
    main()


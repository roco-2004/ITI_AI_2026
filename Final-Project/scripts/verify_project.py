"""Run fast, offline publication-readiness checks for committed project assets."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent

REQUIRED_FILES = (
    "README.md",
    ".env.example",
    "docker-compose.yml",
    "notebooks/house_price_model.ipynb",
    "notebooks/data/README.md",
    "models/house_price.pkl",
    "models/locations.json",
    "models/model_metadata.json",
    "backend/app/main.py",
    "backend/requirements.txt",
    "backend/Dockerfile",
    "frontend/package.json",
    "frontend/Dockerfile",
    "docs/data-audit.md",
    "docs/model-report.md",
    "docs/screenshots/home.png",
    "docs/screenshots/prediction-form.png",
    "docs/screenshots/prediction-result.png",
    "docs/screenshots/swagger.png",
    "scripts/smoke_test.py",
)

METADATA_KEYS = {
    "training_timestamp_utc",
    "dataset_sha256",
    "dataset_shape",
    "feature_names",
    "input_schema",
    "target_definition",
    "currency",
    "random_seed",
    "split_sizes",
    "model_name",
    "model_parameters",
    "test_metrics",
    "cross_validation",
    "versions",
}


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(f"{'PASS' if condition else 'FAIL'}: {message}")
    if not condition:
        failures.append(message)


def git_candidates() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def markdown_links(path: Path) -> list[Path]:
    links: list[Path] = []
    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = target.strip().split("#", 1)[0]
        if not target or re.match(r"^(?:https?://|mailto:)", target):
            continue
        links.append((path.parent / unquote(target)).resolve())
    return links


def main() -> int:
    failures: list[str] = []

    for relative in REQUIRED_FILES:
        check((PROJECT_ROOT / relative).is_file(), f"required file exists: {relative}", failures)

    metadata = json.loads((PROJECT_ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    check(
        metadata.keys() >= METADATA_KEYS,
        "model metadata contains every required field",
        failures,
    )
    check(metadata.get("currency") == "INR", "model currency is INR", failures)
    check(
        len(metadata.get("feature_names", [])) == 11,
        "model metadata declares 11 inputs",
        failures,
    )

    locations = json.loads((PROJECT_ROOT / "models/locations.json").read_text(encoding="utf-8"))
    check(
        isinstance(locations, list) and "Other" in locations,
        "locations include the Other fallback",
        failures,
    )

    notebook_path = PROJECT_ROOT / "notebooks/house_price_model.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    check(bool(code_cells), "notebook JSON contains code cells", failures)
    check(
        all(cell.get("execution_count") is not None for cell in code_cells),
        "all code cells were executed",
        failures,
    )
    check(not error_outputs, "notebook contains no saved error outputs", failures)

    model_size = (PROJECT_ROOT / "models/house_price.pkl").stat().st_size
    check(model_size < 50 * 1024 * 1024, "model artifact is below 50 MiB", failures)

    candidates = git_candidates()
    forbidden = [
        item
        for item in candidates
        if (item.startswith("Final-Project/notebooks/data/") and not item.endswith("/README.md"))
        or "/node_modules/" in item
        or item.endswith((".csv", ".zip", ".7z", ".npy", ".npz", ".parquet", ".parq"))
        or Path(item).name == ".env"
    ]
    check(
        not forbidden,
        f"Git candidates exclude raw data, dependencies, and .env files: {forbidden}",
        failures,
    )

    broken_links: list[str] = []
    for readme in (REPOSITORY_ROOT / "README.md", PROJECT_ROOT / "README.md"):
        if not readme.exists():
            continue
        broken_links.extend(str(link) for link in markdown_links(readme) if not link.exists())
    check(not broken_links, f"local README links resolve: {broken_links}", failures)

    print(f"\nChecked {len(REQUIRED_FILES)} required files and {len(candidates)} Git candidates.")
    if failures:
        print(f"Verification failed with {len(failures)} issue(s).", file=sys.stderr)
        return 1
    print("Project verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

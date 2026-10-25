from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PACKAGE_ROOT / "PACKAGE_MANIFEST.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def current_files() -> dict[str, str]:
    result = {}
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        if path.name in {".DS_Store"} or "__pycache__" in path.parts:
            continue
        result[path.relative_to(PACKAGE_ROOT).as_posix()] = digest(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or verify the supplementary package manifest.")
    parser.add_argument("--write", action="store_true", help="Write PACKAGE_MANIFEST.json.")
    args = parser.parse_args()

    if args.write:
        payload = {
            "algorithm": "sha256",
            "files": current_files(),
        }
        MANIFEST_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {MANIFEST_PATH} with {len(payload['files'])} files.")
        return

    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
    actual = current_files()
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(expected) & set(actual) if expected[path] != actual[path]
        )
        raise SystemExit(
            f"Manifest mismatch: missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    print(f"Verified {len(actual)} package files.")


if __name__ == "__main__":
    main()

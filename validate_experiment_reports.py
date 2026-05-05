import argparse
import glob
import json
from pathlib import Path

from experiment_report import validate_standard_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate benchmark reports that contain a standard_report block."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="JSON reports to validate. Defaults to benchmark_runs/*.json.",
    )
    parser.add_argument(
        "--require-standard",
        action="store_true",
        help="Fail reports that do not contain standard_report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = args.paths or sorted(glob.glob("benchmark_runs/*.json"))
    failures = 0
    checked = 0
    skipped = 0

    for raw_path in paths:
        path = Path(raw_path)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"FAIL {path}: could not parse JSON: {exc}")
            failures += 1
            continue

        standard_report = report.get("standard_report")
        if standard_report is None:
            if args.require_standard:
                print(f"FAIL {path}: missing standard_report")
                failures += 1
            else:
                skipped += 1
            continue

        errors = validate_standard_report(standard_report)
        if errors:
            print(f"FAIL {path}:")
            for error in errors:
                print(f"  - {error}")
            failures += 1
        else:
            print(f"OK   {path}")
            checked += 1

    print(f"checked={checked} skipped={skipped} failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

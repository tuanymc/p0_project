"""Aggregate P0 diagnostic outputs into a markdown report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_p0_diagnostic_report(out_dir: Path) -> Path:
    """Aggregate available CSVs and reports into one diagnostic markdown file."""
    logger.info("Generating P0 diagnostic report in %s", out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "p0_diagnostic_report.md"
    sections = ["# P0 Diagnostic Report", ""]
    for csv_path in sorted(Path("results/tables").glob("*.csv")):
        sections.append(f"## {csv_path.name}")
        sections.append(f"Source: `{csv_path}`")
        sections.append("")
    for report_path in sorted(Path("results/reports").glob("*.md")):
        if report_path.name == path.name:
            continue
        sections.append(f"## {report_path.name}")
        sections.append(report_path.read_text(encoding="utf-8"))
        sections.append("")
    path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Generated report at %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate P0 diagnostic report")
    parser.add_argument("--out", type=Path, default=Path("results/reports/"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    generate_p0_diagnostic_report(args.out)


if __name__ == "__main__":
    main()

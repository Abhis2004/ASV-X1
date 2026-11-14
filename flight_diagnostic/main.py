"""
Flight Diagnostic & CSV Converter - Unified Runner (main.py)
-------------------------------------------------------------
Executes DataFlash binary flight log conversion to CSV and comprehensive
incident diagnostics all at once in a single command.

Usage:
    python flight_diagnostic/main.py
    python flight_diagnostic/main.py "C:\\path\\to\\flight.bin"
    python flight_diagnostic/main.py "C:\\path\\to\\flight.bin" -o ./flight_results
    python flight_diagnostic/main.py --show-report
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Any

# Ensure local package directory is on sys.path
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from convert_bin_to_csv import parse_dataflash_log
    from flight_diagnostics import FlightDiagnosticsAnalyzer
except ImportError:
    from .convert_bin_to_csv import parse_dataflash_log
    from .flight_diagnostics import FlightDiagnosticsAnalyzer


def run_all(
    bin_path: str = r"C:\Users\araj8\Downloads\dataflash_log_104.bin",
    output_dir: Optional[str] = None,
    message_types: Optional[List[str]] = None,
    engine: str = 'auto',
    verbose: bool = True,
    show_report: bool = False
) -> Dict[str, Any]:
    """
    Execute DataFlash binary log parsing, full CSV conversion, and comprehensive
    flight diagnostics all at once.

    Args:
        bin_path: Path to the .bin flight log file.
        output_dir: Parent output folder (defaults to `<bin_stem>_flight_analysis`).
        message_types: Optional list of message types to export as CSV (e.g. ['GPS', 'ATT']).
                       If None, all message types are exported.
        engine: 'auto' (pymavlink with automatic fallback), 'pymavlink', or 'custom'.
        verbose: Print progress and diagnostic summary to console.
        show_report: Print complete incident investigation text report to console.

    Returns:
        Dict containing summary of created files and diagnostic findings.
    """
    bin_file = Path(bin_path).resolve()
    if not bin_file.exists():
        raise FileNotFoundError(f"Flight log file not found: {bin_file}")

    # Set up unified output directory
    if output_dir is None:
        parent_out = bin_file.parent / f"{bin_file.stem}_flight_analysis"
    else:
        parent_out = Path(output_dir).resolve()

    csv_out_dir = parent_out / "csv"
    diag_out_dir = parent_out / "diagnostics"
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    diag_out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("PIXHAWK / ARDUPILOT FLIGHT LOG ANALYZER & CSV CONVERTER")
        print("=" * 80)
        print(f"Target Binary Log:   {bin_file}")
        print(f"Parent Output Folder:{parent_out}")
        print(f"  - CSV Folder:      {csv_out_dir}")
        print(f"  - Diagnostics:     {diag_out_dir}")
        print("-" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Parse DataFlash Binary Log
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 1/3] Parsing DataFlash telemetry streams...")

    columns_by_type, rows_by_type, parser_used = parse_dataflash_log(
        str(bin_file),
        target_types=None,  # Parse all streams so diagnostics has complete context
        engine=engine,
        verbose=verbose
    )

    total_records = sum(len(r) for r in rows_by_type.values())
    if verbose:
        print(f"  Engine Used:       {parser_used}")
        print(f"  Parsed Streams:    {len(rows_by_type)} message types ({total_records:,} total rows)")

    # -------------------------------------------------------------------------
    # Step 2: Convert and Export CSV Files
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2/3] Exporting telemetry messages to CSV format...")

    target_types_set = set(t.upper() for t in message_types) if message_types else None
    exported_csvs: Dict[str, str] = {}

    import csv
    for name, rows in sorted(rows_by_type.items()):
        if not rows:
            continue
        if target_types_set and name not in target_types_set:
            continue

        cols = columns_by_type.get(name, [f'Field_{i+1}' for i in range(len(rows[0]))])
        target_file = csv_out_dir / f"{name}.csv"
        with open(target_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        exported_csvs[name] = str(target_file)

    if verbose:
        print(f"  Successfully wrote {len(exported_csvs)} CSV files into: {csv_out_dir}")

    # -------------------------------------------------------------------------
    # Step 3: Run Comprehensive Flight Diagnostics
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 3/3] Executing incident diagnostics & timeline correlation...")

    analyzer = FlightDiagnosticsAnalyzer(
        columns_by_type=columns_by_type,
        rows_by_type=rows_by_type,
        metadata={'bin_path': str(bin_file)}
    )

    diagnostic_files = analyzer.export_diagnostics(str(diag_out_dir))
    diag_summary = analyzer.generate_incident_diagnosis()

    if verbose:
        print("  Generated Diagnostic Deliverables:")
        for k, p in diagnostic_files.items():
            print(f"    * {k:<20} -> {Path(p).name}")

        print("\n" + "=" * 80)
        print("INCIDENT INVESTIGATION SUMMARY")
        print("=" * 80)
        inc = diag_summary['incident_detection']
        meta = diag_summary['metadata']
        print(f"Vehicle Type:        {meta.get('vehicle_type')} (Frame: {meta.get('frame_type')})")
        print(f"Flight Duration:     {meta.get('flight_duration_s')}s airborne (Total Log: {meta.get('log_duration_s')}s)")
        print(f"Primary Finding:     {inc.get('primary_diagnosis')}")
        print(f"Earliest Alert:      {inc.get('earliest_warning_text')} at Flight Time +{inc.get('earliest_warning_flight_s')}s")
        print(f"Impact Shock:        {inc.get('impact_vibration_spike_z'):.2f} m/s^2 at Flight Time +{inc.get('impact_flight_s')}s")

        print("\nRanked Hypotheses:")
        for h in diag_summary['ranked_hypotheses']:
            print(f"  [{h['rank']}] {h['title']} (Confidence: {h['confidence']})")

        print("\nRecommended Inspections:")
        for i, check in enumerate(diag_summary['recommended_physical_checks'][:3], 1):
            print(f"  {i}. {check}")

        print("\nFull Report Available At:")
        print(f"  {diagnostic_files['flight_diagnosis']}")
        print("=" * 80)

    if show_report:
        report_path = Path(diagnostic_files['flight_diagnosis'])
        if report_path.exists():
            print("\n" + report_path.read_text(encoding='utf-8'))

    return {
        'csv_files': exported_csvs,
        'diagnostic_files': diagnostic_files,
        'summary': diag_summary,
        'parent_dir': str(parent_out)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Flight Diagnostics & CSV Converter for Pixhawk / ArduPilot DataFlash Logs"
    )
    parser.add_argument(
        "bin_file",
        nargs="?",
        default=r"C:\Users\araj8\Downloads\dataflash_log_104.bin",
        help="Path to DataFlash .bin flight log file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Parent directory for CSV and diagnostic outputs (default: <bin_stem>_flight_analysis)",
    )
    parser.add_argument(
        "-t", "--types",
        nargs="+",
        help="Filter specific message types for CSV export (e.g. -t GPS ATT RCOU)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pymavlink", "custom"],
        default="auto",
        help="DataFlash parsing engine (default: auto)",
    )
    parser.add_argument(
        "--show-report",
        action="store_true",
        help="Print the complete incident investigation text report to the terminal",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        run_all(
            bin_path=args.bin_file,
            output_dir=args.output_dir,
            message_types=args.types,
            engine=args.engine,
            verbose=not args.quiet,
            show_report=args.show_report
        )
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

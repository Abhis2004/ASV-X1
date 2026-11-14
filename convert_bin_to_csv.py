"""
Pixhawk / ArduPilot DataFlash (.bin) to CSV Converter & Flight Diagnostics
--------------------------------------------------------------------------
Converts ArduPilot / Pixhawk DataFlash binary flight logs (.bin) into CSV format.
Supports exporting all message types into individual CSV files, filtering specific
message types (e.g. GPS, ATT, IMU, BAT), listing message types, and executing
evidence-based flight incident diagnostics.

Features:
- Dual-engine architecture:
  * Established ArduPilot parser: pymavlink.DFReader (tried first when available)
  * Robust custom recovery parser: DataFlashLogReader (zero external dependencies,
    handles corrupted sectors, un-terminated strings, and format mismatches).
- Automatic engine fallback with transparent parser diagnostics.
- Full support for all 21 ArduPilot DataFlash data types (b, B, h, H, i, I, f, d,
  n, N, Z, c, C, e, E, L, M, q, Q, g, a).
- Accurate units, scaling multipliers, and column labels preservation.
- Optional flight diagnostics (--analyze) generating:
  1. flight_summary.json
  2. event_timeline.csv
  3. motor_analysis.csv
  4. battery_analysis.csv
  5. vibration_analysis.csv
  6. gps_ekf_analysis.csv
  7. flight_diagnosis.txt
- Can be run from CLI or imported as a Python module.
"""

import os
import sys
import struct
import re
import csv
import io
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try importing pymavlink for standard DataFlash parsing
try:
    from pymavlink import mavutil, DFReader
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False


# ArduPilot DataFlash Format characters to byte size
FORMAT_SIZES: Dict[str, int] = {
    'b': 1,   # int8_t
    'B': 1,   # uint8_t
    'h': 2,   # int16_t
    'H': 2,   # uint16_t
    'i': 4,   # int32_t
    'I': 4,   # uint32_t
    'f': 4,   # float (32-bit)
    'd': 8,   # double (64-bit)
    'n': 4,   # char[4]
    'N': 16,  # char[16]
    'Z': 64,  # char[64]
    'c': 2,   # int16_t * 100
    'C': 2,   # uint16_t * 100
    'e': 4,   # int32_t * 100
    'E': 4,   # uint32_t * 100
    'L': 4,   # int32_t latitude/longitude (* 1e-7)
    'M': 1,   # uint8_t flight mode
    'q': 8,   # int64_t
    'Q': 8,   # uint64_t (usually TimeUS)
    'g': 2,   # 16-bit float
    'a': 64,  # int16_t[32] array
}

# Format character mapping to struct format, multiplier, and python type
FORMAT_TO_STRUCT: Dict[str, Tuple[str, Optional[float], type]] = {
    'b': ('b', None, int),
    'B': ('B', None, int),
    'h': ('h', None, int),
    'H': ('H', None, int),
    'i': ('i', None, int),
    'I': ('I', None, int),
    'f': ('f', None, float),
    'd': ('d', None, float),
    'n': ('4s', None, str),
    'N': ('16s', None, str),
    'Z': ('64s', None, str),
    'c': ('h', 0.01, float),
    'C': ('H', 0.01, float),
    'e': ('i', 0.01, float),
    'E': ('I', 0.01, float),
    'L': ('i', 1e-07, float),
    'M': ('b', None, int),
    'q': ('q', None, int),
    'Q': ('Q', None, int),
    'g': ('e', None, float),
    'a': ('32h', None, list),
}

# Message header bytes for DataFlash
HEAD1 = 0xA3
HEAD2 = 0x95
FMT_TYPE = 0x80


def _null_term(b: bytes) -> str:
    """Extract null-terminated string from bytes."""
    idx = b.find(b'\x00')
    if idx != -1:
        b = b[:idx]
    return b.decode('latin1', errors='ignore')


def _clean_name(s: str) -> str:
    """Extract clean alphanumeric name for message type or field."""
    m = re.match(r'^[A-Za-z0-9_]+', s.strip())
    return m.group(0) if m else 'UNKNOWN'


def _clean_columns(col_bytes: bytes, num_fields: int) -> List[str]:
    """
    Parse and clean column label strings from FMT messages.
    Handles un-terminated strings and buffer noise common in embedded logs.
    """
    text = _null_term(col_bytes)
    raw_cols = text.split(',')
    cleaned_cols = []
    for c in raw_cols:
        m = re.match(r'^[A-Za-z0-9_]+', c.strip())
        if m:
            cleaned_cols.append(m.group(0))
        else:
            break

    # Truncate if there are more column labels than format fields
    if len(cleaned_cols) > num_fields:
        cleaned_cols = cleaned_cols[:num_fields]

    # Fill default names if columns are missing
    while len(cleaned_cols) < num_fields:
        cleaned_cols.append(f'Field_{len(cleaned_cols) + 1}')

    # Ensure unique column names
    seen = {}
    unique_cols = []
    for col in cleaned_cols:
        if col in seen:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            unique_cols.append(col)

    return unique_cols


def _compile_format(fmt_str: str) -> Tuple[Optional[str], List[Optional[float]], List[type]]:
    """Compile a format string into struct pattern, multipliers, and python types."""
    struct_str = '<'
    multipliers = []
    types = []
    for c in fmt_str:
        if c in FORMAT_TO_STRUCT:
            s, mul, t = FORMAT_TO_STRUCT[c]
            struct_str += s
            multipliers.append(mul)
            types.append(t)
        else:
            return None, [], []
    return struct_str, multipliers, types


class DataFlashLogReader:
    """
    Streaming reader for ArduPilot / Pixhawk DataFlash .bin log files.
    Pure Python recovery parser capable of recovering records from corrupted logs.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.formats: Dict[int, Dict[str, Any]] = {}
        self.stats: Dict[str, int] = {
            'records_parsed': 0,
            'skipped_bytes': 0,
            'malformed_records': 0,
            'format_count': 0
        }
        # Seed FMT message definition (type 0x80)
        self.formats[FMT_TYPE] = {
            'len': 89,
            'name': 'FMT',
            'format': 'BBnNZ',
            'columns': ['Type', 'Length', 'Name', 'Format', 'Columns'],
            'struct': '<BB4s16s64s',
            'mults': [None, None, None, None, None],
            'types': [int, int, str, str, str],
        }

    def _parse_fmt(self, body: bytes) -> None:
        """Parse a FMT message definition."""
        ftype, flen, fname, ffmt, fcols = struct.unpack('<BB4s16s64s', body[:86])
        clean_fname = _clean_name(_null_term(fname))
        raw_ffmt = _null_term(ffmt)

        # In ArduPilot logs, flen is the ground-truth total length (including 3 header bytes)
        target_payload = flen - 3

        # Match format characters until cumulative byte size equals payload length
        cur_sz = 0
        valid_fmt = ''
        for c in raw_ffmt:
            sz = FORMAT_SIZES.get(c, 0)
            if cur_sz + sz <= target_payload:
                valid_fmt += c
                cur_sz += sz
            else:
                break

        # Clean column labels to match the valid format field count
        num_fields = len(valid_fmt)
        cols = _clean_columns(fcols, num_fields)

        s_str, mults, types = _compile_format(valid_fmt)

        self.formats[ftype] = {
            'len': flen,
            'name': clean_fname,
            'format': valid_fmt,
            'columns': cols,
            'struct': s_str,
            'mults': mults,
            'types': types,
        }

    def scan_message_counts(self) -> Dict[str, int]:
        """Scan the file and return a dictionary of {msg_type: row_count}."""
        counts: Dict[str, int] = {}
        with open(self.filepath, 'rb') as f:
            data = f.read()

        offset = 0
        data_len = len(data)
        while offset < data_len - 3:
            if data[offset] == HEAD1 and data[offset + 1] == HEAD2:
                mtype = data[offset + 2]
                if mtype in self.formats:
                    fmt = self.formats[mtype]
                    mlen = fmt['len']
                    if offset + mlen <= data_len:
                        if mtype == FMT_TYPE:
                            self._parse_fmt(data[offset + 3:offset + mlen])
                        else:
                            name = fmt['name']
                            counts[name] = counts.get(name, 0) + 1
                        offset += mlen
                        continue
            offset += 1
        return counts

    def parse(self, target_types: Optional[set] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[List[Any]]]]:
        """
        Parse binary file and extract rows.
        Returns:
            columns_by_type: {msg_name: [col1, col2, ...]}
            rows_by_type: {msg_name: [[val1, val2, ...], ...]}
        """
        columns_by_type: Dict[str, List[str]] = {}
        rows_by_type: Dict[str, List[List[Any]]] = {}

        with open(self.filepath, 'rb') as f:
            data = f.read()

        offset = 0
        data_len = len(data)
        skipped_bytes = 0
        malformed = 0
        parsed_count = 0

        while offset < data_len - 3:
            if data[offset] == HEAD1 and data[offset + 1] == HEAD2:
                mtype = data[offset + 2]
                if mtype in self.formats:
                    fmt = self.formats[mtype]
                    mlen = fmt['len']
                    if offset + mlen <= data_len:
                        body = data[offset + 3:offset + mlen]
                        if mtype == FMT_TYPE:
                            self._parse_fmt(body)
                        else:
                            name = fmt['name']
                            if target_types is None or name in target_types:
                                if fmt['struct'] is not None:
                                    try:
                                        vals = struct.unpack(fmt['struct'], body)
                                        processed = []
                                        val_idx = 0
                                        for mul, t in zip(fmt['mults'], fmt['types']):
                                            if t is list:  # int16_t[32] array ('a')
                                                arr = list(vals[val_idx:val_idx + 32])
                                                val_idx += 32
                                                processed.append(';'.join(str(x) for x in arr))
                                            else:
                                                v = vals[val_idx]
                                                val_idx += 1
                                                if t is str:
                                                    if isinstance(v, bytes):
                                                        v = _null_term(v)
                                                elif mul is not None:
                                                    v = round(v * mul, 7)
                                                processed.append(v)

                                        if name not in rows_by_type:
                                            rows_by_type[name] = []
                                            columns_by_type[name] = fmt['columns']
                                        rows_by_type[name].append(processed)
                                        parsed_count += 1
                                    except Exception:
                                        malformed += 1
                                        # Resynchronize by 1 byte on unpack failure
                                        offset += 1
                                        continue
                        offset += mlen
                        continue
            offset += 1
            skipped_bytes += 1

        self.stats['records_parsed'] = parsed_count
        self.stats['skipped_bytes'] = skipped_bytes
        self.stats['malformed_records'] = malformed
        self.stats['format_count'] = len(self.formats)

        return columns_by_type, rows_by_type


def _parse_with_pymavlink(
    bin_path: str,
    target_types: Optional[set] = None,
    silence_c_indexer: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[Any]]]]:
    """
    Attempt parsing with pymavlink.DFReader.
    Optionally silences dfindexer C extension stdout/stderr to keep terminal output clean.
    """
    if not PYMAVLINK_AVAILABLE:
        raise ImportError("pymavlink is not installed in the current Python environment.")

    old_stdout_fd = None
    old_stderr_fd = None

    if silence_c_indexer:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:
            old_stdout_fd = None
            old_stderr_fd = None

    try:
        log = mavutil.mavlink_connection(bin_path, zero_time_base=False)
        columns_by_type: Dict[str, List[str]] = {}
        rows_by_type: Dict[str, List[List[Any]]] = {}

        try:
            while True:
                m = log.recv_msg()
                if m is None:
                    break
                mtype = m.get_type()
                if target_types is not None and mtype not in target_types:
                    continue

                if mtype not in columns_by_type:
                    columns_by_type[mtype] = m.get_fieldnames()
                    rows_by_type[mtype] = []

                # Extract fields in column order
                fieldnames = columns_by_type[mtype]
                row = [getattr(m, f) for f in fieldnames]
                rows_by_type[mtype].append(row)
        finally:
            if hasattr(log, 'close'):
                log.close()

        return columns_by_type, rows_by_type
    finally:
        if old_stdout_fd is not None:
            try:
                os.dup2(old_stdout_fd, 1)
                os.close(old_stdout_fd)
            except Exception:
                pass
        if old_stderr_fd is not None:
            try:
                os.dup2(old_stderr_fd, 2)
                os.close(old_stderr_fd)
            except Exception:
                pass


def parse_dataflash_log(
    bin_path: str,
    target_types: Optional[set] = None,
    engine: str = 'auto',
    verbose: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, List[List[Any]]], str]:
    """
    Parse DataFlash log using the selected engine ('auto', 'pymavlink', 'custom').
    Returns:
        (columns_by_type, rows_by_type, parser_used)
    """
    engine = engine.lower()
    if engine not in ('auto', 'pymavlink', 'custom'):
        raise ValueError(f"Invalid engine '{engine}'. Must be one of: 'auto', 'pymavlink', 'custom'.")

    # 1. Attempt pymavlink if requested
    if engine in ('pymavlink', 'auto') and PYMAVLINK_AVAILABLE:
        try:
            if verbose and engine == 'pymavlink':
                print("Parsing with pymavlink.DFReader...")
            cols, rows = _parse_with_pymavlink(bin_path, target_types=target_types, silence_c_indexer=True)
            return cols, rows, "pymavlink.DFReader"
        except Exception as e:
            if engine == 'pymavlink':
                raise RuntimeError(f"pymavlink parser failed: {e}")
            if verbose:
                print(f"Notice: Standard pymavlink parser encountered error ({type(e).__name__}: {e}).")
                print("Switching seamlessly to custom recovery parser...")

    # 2. Use custom recovery parser
    reader = DataFlashLogReader(bin_path)
    cols, rows = reader.parse(target_types=target_types)
    parser_label = "DataFlashLogReader (Custom Recovery Engine)"
    if engine == 'auto' and PYMAVLINK_AVAILABLE:
        parser_label += " [Fallback]"
    return cols, rows, parser_label


def convert_bin_to_csv(
    bin_path: str,
    output_dir: Optional[str] = None,
    message_types: Optional[List[str]] = None,
    verbose: bool = True,
    engine: str = 'auto',
    analyze: bool = False,
    diag_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Convert a Pixhawk / ArduPilot DataFlash .bin log file to CSV files.

    Args:
        bin_path: Path to the input .bin log file.
        output_dir: Output directory where CSV files will be stored.
                    Defaults to `<bin_filename>_csv` in the same directory.
        message_types: Optional list of message types to export (e.g. ['GPS', 'ATT']).
                       If None, all message types are exported.
        verbose: Whether to print progress messages.
        engine: 'auto' (default: try pymavlink with fallback), 'pymavlink', or 'custom'.
        analyze: If True, execute flight diagnostics and export diagnostic reports.
        diag_dir: Directory for diagnostics reports (defaults to `<bin_filename>_diagnostics`).

    Returns:
        Dict mapping message type name to output CSV file path.
    """
    bin_file = Path(bin_path).resolve()
    if not bin_file.exists():
        raise FileNotFoundError(f"Binary log file not found: {bin_file}")

    if output_dir is None:
        out_path = bin_file.parent / f"{bin_file.stem}_csv"
    else:
        out_path = Path(output_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)

    target_types_set = set(t.upper() for t in message_types) if message_types else None

    # If running diagnostics, ensure critical diagnostic message streams are parsed
    parse_targets = None
    if target_types_set:
        if analyze:
            diagnostic_critical_types = {'MSG', 'MODE', 'ARM', 'ERR', 'EV', 'CTUN', 'BARO', 'VIBE', 'RCOU', 'POWR', 'GPS', 'PARM'}
            parse_targets = target_types_set | diagnostic_critical_types
        else:
            parse_targets = target_types_set

    if verbose:
        print(f"Reading DataFlash log: {bin_file}")
        print(f"Output directory:      {out_path}")
        if target_types_set:
            print(f"Filtering for message types: {', '.join(sorted(target_types_set))}")

    # Parse log with selected engine
    columns_by_type, rows_by_type, parser_used = parse_dataflash_log(
        str(bin_file),
        target_types=parse_targets,
        engine=engine,
        verbose=verbose
    )

    if verbose:
        print(f"Parser engine used:    {parser_used}")

    created_files: Dict[str, str] = {}
    total_messages = 0

    for name, rows in sorted(rows_by_type.items()):
        if not rows:
            continue
        # If user explicitly filtered message types, only export those to CSV
        if target_types_set and name not in target_types_set:
            continue

        cols = columns_by_type.get(name, [f'Field_{i+1}' for i in range(len(rows[0]))])
        csv_file = out_path / f"{name}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)

        created_files[name] = str(csv_file)
        total_messages += len(rows)
        if verbose:
            print(f"  Exported {name:>6}: {len(rows):>6} rows -> {csv_file.name}")

    if verbose:
        print(f"\nCompleted! Generated {len(created_files)} CSV files ({total_messages:,} total rows).")
        print(f"Destination folder:    {out_path}")

    # Run Flight Diagnostics if requested
    if analyze:
        try:
            from flight_diagnostics import run_flight_diagnostics
            d_dir = diag_dir or (bin_file.parent / f"{bin_file.stem}_diagnostics")
            run_flight_diagnostics(
                bin_path=str(bin_file),
                output_dir=str(d_dir),
                columns_by_type=columns_by_type,
                rows_by_type=rows_by_type,
                verbose=verbose
            )
        except Exception as e:
            print(f"Warning: Diagnostics execution encountered an error: {e}", file=sys.stderr)

    return created_files


def list_bin_message_types(bin_path: str, engine: str = 'auto') -> Dict[str, int]:
    """Scan and list all message types and their row counts in a .bin log file."""
    cols, rows, _ = parse_dataflash_log(bin_path, engine=engine, verbose=False)
    return {k: len(v) for k, v in rows.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pixhawk / ArduPilot DataFlash .bin flight logs to CSV & Run Flight Diagnostics."
    )
    parser.add_argument(
        "bin_file",
        nargs="?",
        default=r"C:\Users\araj8\Downloads\dataflash_log_104.bin",
        help="Path to the .bin flight log file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the CSV files (default: <filename>_csv)",
    )
    parser.add_argument(
        "-t", "--types",
        nargs="+",
        help="Filter specific message types (e.g. -t GPS ATT IMU BAT)",
    )
    parser.add_argument(
        "-l", "--list-types",
        action="store_true",
        help="List all message types and their row counts without exporting CSVs",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run comprehensive flight log diagnostics and generate incident report",
    )
    parser.add_argument(
        "--diag-dir",
        default=None,
        help="Directory to save diagnostics reports (default: <filename>_diagnostics)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pymavlink", "custom"],
        default="auto",
        help="DataFlash parsing engine (default: auto)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    if not os.path.exists(args.bin_file):
        print(f"Error: Input file does not exist: {args.bin_file}", file=sys.stderr)
        sys.exit(1)

    if args.list_types:
        print(f"Scanning message types in: {args.bin_file}")
        counts = list_bin_message_types(args.bin_file, engine=args.engine)
        print(f"\nFound {len(counts)} message types:")
        print(f"{'Type':<10} {'Rows':>10}")
        print("-" * 22)
        for msg_type, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{msg_type:<10} {count:>10,}")
        print("-" * 22)
        print(f"{'Total':<10} {sum(counts.values()):>10,}")
        return

    try:
        convert_bin_to_csv(
            bin_path=args.bin_file,
            output_dir=args.output_dir,
            message_types=args.types,
            verbose=not args.quiet,
            engine=args.engine,
            analyze=args.analyze,
            diag_dir=args.diag_dir,
        )
    except Exception as e:
        print(f"Error processing log: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

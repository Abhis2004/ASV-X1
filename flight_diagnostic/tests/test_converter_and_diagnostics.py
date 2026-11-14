"""
Unit and Integration Test Suite for DataFlash Converter & Flight Diagnostics
-----------------------------------------------------------------------------
Validates:
1. All 21 DataFlash format characters, field sizes, multipliers, and struct formats.
2. FMT parsing, length reconciliation, and column label sanitization.
3. Sector corruption recovery and byte resynchronization.
4. Diagnostics metrics (motor spread, saturation, missing battery handling).
5. Full end-to-end conversion and diagnostics on real DataFlash log.
"""

import os
import sys
import unittest
import struct
import json
import tempfile
import shutil
from pathlib import Path

# Add package directory to path
PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from convert_bin_to_csv import (
    DataFlashLogReader,
    convert_bin_to_csv,
    list_bin_message_types,
    parse_dataflash_log,
    FORMAT_SIZES,
    FORMAT_TO_STRUCT,
    _compile_format,
    _clean_columns,
    HEAD1,
    HEAD2
)
from flight_diagnostics import FlightDiagnosticsAnalyzer, run_flight_diagnostics
from main import run_all


class TestFormatSpecifiers(unittest.TestCase):
    """Test DataFlash format specifiers, struct compilation, and size definitions."""

    def test_all_format_characters_defined(self):
        """Verify all 21 ArduPilot DataFlash format specifiers are mapped."""
        expected_chars = ['b', 'B', 'h', 'H', 'i', 'I', 'f', 'd', 'n', 'N', 'Z',
                          'c', 'C', 'e', 'E', 'L', 'M', 'q', 'Q', 'g', 'a']
        for c in expected_chars:
            self.assertIn(c, FORMAT_SIZES, f"Missing format size for '{c}'")
            self.assertIn(c, FORMAT_TO_STRUCT, f"Missing struct mapping for '{c}'")

    def test_compile_format_struct(self):
        """Test compiling complex format string into struct pattern."""
        fmt_str = "Qfffff"
        s_str, mults, types = _compile_format(fmt_str)
        self.assertEqual(s_str, "<Qfffff")
        self.assertEqual(struct.calcsize(s_str), 28)
        self.assertEqual(len(mults), 6)
        self.assertEqual(len(types), 6)

    def test_scaled_format_characters(self):
        """Verify centi-units (c, C, e, E) and lat/lng (L) multipliers."""
        _, mults_c, _ = _compile_format("c")
        self.assertEqual(mults_c[0], 0.01)

        _, mults_C, _ = _compile_format("C")
        self.assertEqual(mults_C[0], 0.01)

        _, mults_L, _ = _compile_format("L")
        self.assertEqual(mults_L[0], 1e-07)

    def test_array_format_character(self):
        """Verify 'a' format specifier maps to 32 int16 values (64 bytes)."""
        s_str, mults, types = _compile_format("a")
        self.assertEqual(s_str, "<32h")
        self.assertEqual(struct.calcsize(s_str), 64)
        self.assertIs(types[0], list)


class TestFmtParsingAndSanitization(unittest.TestCase):
    """Test FMT message parsing and column label cleaning."""

    def test_column_sanitization_and_padding(self):
        """Verify column names are cleaned, padded, or truncated to match field count."""
        raw_bytes = b"TimeUS,Roll,Pitch,Yaw\x00\xff\xfe\x00garbage"
        cols = _clean_columns(raw_bytes, num_fields=4)
        self.assertEqual(cols, ["TimeUS", "Roll", "Pitch", "Yaw"])

        cols_padded = _clean_columns(b"TimeUS,Val1\x00", num_fields=4)
        self.assertEqual(cols_padded, ["TimeUS", "Val1", "Field_3", "Field_4"])

        cols_truncated = _clean_columns(b"TimeUS,A,B,C,D,E\x00", num_fields=3)
        self.assertEqual(cols_truncated, ["TimeUS", "A", "B"])

    def test_duplicate_column_names_made_unique(self):
        """Ensure duplicate column names receive unique suffixes."""
        raw_bytes = b"TimeUS,Val,Val,Val\x00"
        cols = _clean_columns(raw_bytes, num_fields=4)
        self.assertEqual(cols, ["TimeUS", "Val", "Val_2", "Val_3"])


class TestCorruptDataRecovery(unittest.TestCase):
    """Test parser synchronization and recovery from corrupted binary streams."""

    def test_corrupt_stream_resynchronization(self):
        stream = bytearray()
        fmt_body = struct.pack(
            '<BB4s16s64s',
            144,
            16,
            b'TEST\x00',
            b'QfB\x00',
            b'TimeUS,ValF,ValB\x00'
        )
        stream.extend([HEAD1, HEAD2, 0x80])
        stream.extend(fmt_body)
        stream.extend(b"\x00\x00\x55\xaa\xff\x00\x12\x34\x56\x78")

        test_body_1 = struct.pack('<QfB', 1000000, 12.5, 7)
        stream.extend([HEAD1, HEAD2, 144])
        stream.extend(test_body_1)

        stream.extend([HEAD1, HEAD2, 0xFE, 0x01, 0x02, 0x03])

        test_body_2 = struct.pack('<QfB', 2000000, 25.0, 9)
        stream.extend([HEAD1, HEAD2, 144])
        stream.extend(test_body_2)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
            tf.write(stream)
            tmp_path = tf.name

        try:
            reader = DataFlashLogReader(tmp_path)
            cols, rows = reader.parse()

            self.assertIn('TEST', rows)
            self.assertEqual(len(rows['TEST']), 2)
            self.assertEqual(rows['TEST'][0][0], 1000000)
            self.assertAlmostEqual(rows['TEST'][0][1], 12.5, places=4)
            self.assertEqual(rows['TEST'][0][2], 7)
            self.assertEqual(rows['TEST'][1][0], 2000000)
            self.assertAlmostEqual(rows['TEST'][1][1], 25.0, places=4)
            self.assertEqual(rows['TEST'][1][2], 9)
            self.assertGreater(reader.stats['skipped_bytes'], 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDiagnosticsCalculations(unittest.TestCase):
    """Test flight diagnostics analysis and calculations on synthetic records."""

    def test_missing_battery_never_treated_as_zero(self):
        """Verify missing BAT messages are reported as missing and not fabricated as 0V."""
        cols = {
            'POWR': ['TimeUS', 'Vcc', 'VServo', 'Flags'],
            'MODE': ['TimeUS', 'Mode', 'ModeNum', 'Rsn'],
        }
        rows = {
            'POWR': [[1000000, 4.91, 4.92, 0], [2000000, 4.90, 4.92, 0]],
            'MODE': [[1000000, 5, 5, 1]],
        }

        analyzer = FlightDiagnosticsAnalyzer(columns_by_type=cols, rows_by_type=rows)
        pwr_rows, pwr_summary = analyzer.analyze_power()

        self.assertFalse(pwr_summary['main_battery_logged'])
        self.assertIn('ABSENT', pwr_summary['main_battery_assessment'])
        self.assertIn('sensor missing', pwr_rows[0]['MainBattery_V'])

    def test_motor_spread_and_saturation(self):
        """Verify motor saturation flag and spread calculation."""
        cols = {
            'RCOU': ['TimeUS', 'C1', 'C2', 'C3', 'C4'],
            'MODE': ['TimeUS', 'Mode', 'ModeNum', 'Rsn'],
            'EV': ['TimeUS', 'Id']
        }
        rows = {
            'RCOU': [
                [10000000, 1949, 1380, 1850, 1700],
                [11000000, 1500, 1500, 1500, 1500],
            ],
            'MODE': [[9000000, 5, 5, 1]],
            'EV': [[9500000, 28]]
        }

        analyzer = FlightDiagnosticsAnalyzer(columns_by_type=cols, rows_by_type=rows)
        motor_rows, motor_summary = analyzer.analyze_motors()

        self.assertEqual(len(motor_rows), 2)
        self.assertEqual(motor_rows[0]['MotorSpread_PWM'], 569.0)
        self.assertEqual(motor_rows[0]['Motor1_Saturated'], 1)
        self.assertIn('M1_SATURATED', motor_rows[0]['AnomalyFlags'])
        self.assertIn('HIGH_SPREAD', motor_rows[0]['AnomalyFlags'])

        self.assertEqual(motor_rows[1]['MotorSpread_PWM'], 0.0)
        self.assertEqual(motor_rows[1]['Motor1_Saturated'], 0)


class TestRealLogEndToEnd(unittest.TestCase):
    """Integration test against the supplied Pixhawk flight log."""

    LOG_PATH = r"C:\Users\araj8\Downloads\dataflash_log_104.bin"

    @classmethod
    def setUpClass(cls):
        cls.log_exists = os.path.exists(cls.LOG_PATH)

    def test_unified_run_all(self):
        """Test unified run_all() function generating both CSVs and diagnostics."""
        if not self.log_exists:
            self.skipTest(f"Test log not found at {self.LOG_PATH}")

        temp_out = tempfile.mkdtemp(prefix="test_run_all_")
        try:
            res = run_all(
                bin_path=self.LOG_PATH,
                output_dir=temp_out,
                message_types=['GPS', 'ATT', 'RCOU'],
                verbose=False
            )

            self.assertIn('csv_files', res)
            self.assertIn('diagnostic_files', res)
            self.assertIn('summary', res)

            # Check CSV files
            self.assertIn('GPS', res['csv_files'])
            self.assertTrue(Path(res['csv_files']['GPS']).exists())

            # Check Diagnostic files
            diag_files = res['diagnostic_files']
            for k in ['flight_summary', 'event_timeline', 'motor_analysis', 'flight_diagnosis']:
                self.assertIn(k, diag_files)
                self.assertTrue(Path(diag_files[k]).exists())

            # Check Primary diagnosis
            self.assertIn('Thrust Loss on Motor 1', res['summary']['incident_detection']['primary_diagnosis'])

        finally:
            shutil.rmtree(temp_out, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

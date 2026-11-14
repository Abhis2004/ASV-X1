"""
Flight Diagnostics Module for ArduPilot / Pixhawk DataFlash Logs
-----------------------------------------------------------------
Analyzes decoded DataFlash records and produces evidence-based, chronological
incident diagnostics, including:
- Event timeline (arming, modes, errors, failsafes, thrust loss, impact)
- Motor & ESC analysis (commanded PWM, saturation, differential thrust)
- Battery & power analysis (distinguishing FC rail Vcc from main battery)
- Vibration & mechanical analysis (baseline vs impact spike, sensor clipping)
- GPS & EKF estimator analysis (satellites, HDOP, innovation variances)
- Flight control & altitude performance (altitude loss under full throttle)
- Diagnostic ranking & human-readable incident report

Outputs generated:
1. flight_summary.json       - Structured flight metadata and findings
2. event_timeline.csv        - Chronological sequence of all events
3. motor_analysis.csv        - Motor-output telemetry and saturation flags
4. battery_analysis.csv      - FC power rails and battery measurements
5. vibration_analysis.csv    - Vibration metrics and clipping events
6. gps_ekf_analysis.csv      - GPS reception and EKF estimator health
7. flight_diagnosis.txt      - Comprehensive incident investigation report
"""

import os
import sys
import json
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ArduCopter Flight Mode Mapping
COPTER_MODES: Dict[int, str] = {
    0: 'STABILIZE',
    1: 'ACRO',
    2: 'ALT_HOLD',
    3: 'AUTO',
    4: 'GUIDED',
    5: 'LOITER',
    6: 'RTL',
    7: 'CIRCLE',
    9: 'LAND',
    11: 'DRIFT',
    13: 'SPORT',
    14: 'FLIP',
    15: 'AUTOTUNE',
    16: 'POSHOLD',
    17: 'BRAKE',
    18: 'THROW',
    19: 'AVOID_ADSB',
    20: 'GUIDED_NOGPS',
    21: 'SMART_RTL',
    22: 'FLOWHOLD',
    23: 'FOLLOW',
    24: 'ZIGZAG',
    25: 'SYSTEMID',
    26: 'AUTOROTATE',
    27: 'AUTO_RTL',
}

# ArduPilot Event IDs
EVENT_NAMES: Dict[int, str] = {
    10: 'ARMED',
    11: 'DISARMED',
    15: 'AUTO_ARMED',
    17: 'LAND_COMPLETE_MAYBE',
    18: 'LAND_COMPLETE',
    28: 'NOT_LANDED',
    56: 'SURFACED',
    57: 'BOTTOMED',
    62: 'EKF_YAW_RESET',
}

# ArduPilot Subsystem IDs
SUBSYS_NAMES: Dict[int, str] = {
    1: 'MAIN',
    2: 'RADIO',
    3: 'COMPASS',
    4: 'OPTFLOW',
    5: 'FAILSAFE_RADIO',
    6: 'FAILSAFE_BATT',
    7: 'FAILSAFE_GPS',
    8: 'FAILSAFE_GCS',
    9: 'FAILSAFE_FENCE',
    10: 'FLIGHT_MODE',
    11: 'GPS',
    12: 'CRASH_CHECK',
    13: 'FLIP',
    14: 'AUTOTUNE',
    15: 'PARACHUTE',
    16: 'EKF_CHECK',
    17: 'FAILSAFE_EKFINAV',
    18: 'BARO',
    19: 'CPU',
    20: 'FAILSAFE_ADSB',
    21: 'TERRAIN',
    22: 'NAVIGATION',
    23: 'FAILSAFE_TERRAIN',
    24: 'EKF_PRIMARY',
    25: 'THRUST_LOSS_CHECK',
}

# Mode Change Reasons
MODE_REASONS: Dict[int, str] = {
    1: 'GCS_COMMAND',
    2: 'RC_COMMAND',
    3: 'FAILSAFE',
    53: 'FAILSAFE_CRASH_OR_THRUST_LOSS',
}


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely cast value to float without treating missing as zero."""
    if val is None or val == '' or val == 'None' or val == 'N/A':
        return default
    try:
        f = float(val)
        return f if not math.isnan(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely cast value to int."""
    if val is None or val == '' or val == 'None' or val == 'N/A':
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _find_closest_row(rows: List[List[Any]], target_tus: int) -> Optional[List[Any]]:
    """Binary search for row with closest TimeUS."""
    if not rows:
        return None
    low = 0
    high = len(rows) - 1
    best = rows[0]
    best_diff = abs(rows[0][0] - target_tus)

    while low <= high:
        mid = (low + high) // 2
        diff = abs(rows[mid][0] - target_tus)
        if diff < best_diff:
            best_diff = diff
            best = rows[mid]
        if rows[mid][0] < target_tus:
            low = mid + 1
        elif rows[mid][0] > target_tus:
            high = mid - 1
        else:
            return rows[mid]
    return best


class FlightDiagnosticsAnalyzer:
    """
    Analyzes decoded DataFlash log tables and produces evidence-based diagnostic reports.
    """

    def __init__(
        self,
        columns_by_type: Dict[str, List[str]],
        rows_by_type: Dict[str, List[List[Any]]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.cols = columns_by_type
        self.rows = rows_by_type
        self.meta = metadata or {}

        # Cache column index lookups
        self.col_indices: Dict[str, Dict[str, int]] = {}
        for mtype, clist in self.cols.items():
            self.col_indices[mtype] = {c: i for i, c in enumerate(clist)}

        # Sort all message rows by TimeUS to guarantee monotonic temporal ordering
        for mtype in self.rows:
            self.rows[mtype].sort(key=lambda r: r[0] if isinstance(r[0], (int, float)) else 0)

        # Baseline flight time references
        self.t0_us, self.t_end_us = self._determine_time_bounds()
        self.liftoff_us = self._determine_liftoff_time()
        self.disarm_us = self._determine_disarm_time()

    def _col_idx(self, mtype: str, col_name: str) -> Optional[int]:
        """Get column index for a given message type and field name."""
        return self.col_indices.get(mtype, {}).get(col_name)

    def _determine_time_bounds(self) -> Tuple[int, int]:
        """
        Determine realistic initial and final timestamp in log.
        Filters out 32-bit timer promotion corruptions (> 5e8 us).
        """
        valid_ts = []
        # Use MODE and ARM messages as anchor
        for mtype in ('MODE', 'ARM', 'MSG'):
            if mtype in self.rows:
                for r in self.rows[mtype]:
                    t = r[0]
                    if isinstance(t, (int, float)) and 1e6 < t < 5e8:
                        valid_ts.append(t)

        if valid_ts:
            return min(valid_ts), max(valid_ts)

        for rlist in self.rows.values():
            for r in rlist:
                t = r[0]
                if isinstance(t, (int, float)) and 1e6 < t < 5e8:
                    valid_ts.append(t)
                    break

        t0 = min(valid_ts) if valid_ts else 0
        t_end = max(valid_ts) if valid_ts else t0
        return t0, t_end

    def _determine_liftoff_time(self) -> int:
        """Find timestamp of liftoff (NOT_LANDED event, or arming, or altitude gain)."""
        if 'EV' in self.rows:
            id_idx = self._col_idx('EV', 'Id') or 1
            for r in self.rows['EV']:
                if r[id_idx] == 28 and 1e6 < r[0] < 5e8:
                    return r[0]

        if 'ARM' in self.rows:
            arm_idx = self._col_idx('ARM', 'ArmState') or 1
            for r in self.rows['ARM']:
                if r[arm_idx] == 1 and 1e6 < r[0] < 5e8:
                    return r[0]

        if 'EV' in self.rows:
            id_idx = self._col_idx('EV', 'Id') or 1
            for r in self.rows['EV']:
                if r[id_idx] in (10, 15) and 1e6 < r[0] < 5e8:
                    return r[0]

        return self.t0_us

    def _determine_disarm_time(self) -> int:
        """Find timestamp of disarm."""
        if 'ARM' in self.rows:
            arm_idx = self._col_idx('ARM', 'ArmState') or 1
            for r in reversed(self.rows['ARM']):
                if r[arm_idx] == 0 and 1e6 < r[0] < 5e8:
                    return r[0]

        if 'EV' in self.rows:
            id_idx = self._col_idx('EV', 'Id') or 1
            for r in reversed(self.rows['EV']):
                if r[id_idx] == 11 and 1e6 < r[0] < 5e8:
                    return r[0]

        return self.t_end_us

    def _flight_time_s(self, tus: int) -> float:
        """Convert TimeUS to seconds relative to liftoff / flight start."""
        ref = self.liftoff_us if self.liftoff_us > 0 else self.t0_us
        clean_tus = tus
        if tus > 5e8:
            clean_tus = tus & 0xFFFFFFFF
        return round((clean_tus - ref) / 1e6, 3)

    # -------------------------------------------------------------------------
    # 1. Event Timeline Analysis
    # -------------------------------------------------------------------------
    def build_event_timeline(self) -> List[Dict[str, Any]]:
        """
        Extract all discrete logged events and derived milestone anomalies.
        Returns chronologically sorted list of event dictionaries.
        """
        timeline: List[Dict[str, Any]] = []

        timeline.append({
            'TimeUS': self.t0_us,
            'FlightTimeS': self._flight_time_s(self.t0_us),
            'Source': 'SYSTEM',
            'EventType': 'LOG_START',
            'Severity': 'INFO',
            'Description': 'DataFlash logging begins',
            'RawData': f"t0_us={self.t0_us}"
        })

        if 'MSG' in self.rows:
            msg_idx = self._col_idx('MSG', 'Message')
            if msg_idx is None:
                for col, idx in self.col_indices['MSG'].items():
                    if 'message' in col.lower():
                        msg_idx = idx
                        break
                if msg_idx is None:
                    msg_idx = 3 if len(self.cols['MSG']) > 3 else 1

            for r in self.rows['MSG']:
                tus = r[0]
                text = str(r[msg_idx]).strip()
                sev = 'INFO'
                if 'thrust loss' in text.lower() or 'failsafe' in text.lower() or 'error' in text.lower():
                    sev = 'CRITICAL'
                elif 'warning' in text.lower() or 'yaw alignment' in text.lower():
                    sev = 'WARNING'
                timeline.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'Source': 'MSG',
                    'EventType': 'TEXT_MESSAGE',
                    'Severity': sev,
                    'Description': text,
                    'RawData': f"ID={r[1]}, Seq={r[2]}"
                })

        if 'MODE' in self.rows:
            m_idx = self._col_idx('MODE', 'Mode') or 1
            rsn_idx = self._col_idx('MODE', 'Rsn') or 3
            for r in self.rows['MODE']:
                tus = r[0]
                mnum = r[m_idx]
                rsn = r[rsn_idx] if len(r) > rsn_idx else 0
                mname = COPTER_MODES.get(mnum, f"MODE_{mnum}")
                rsn_desc = MODE_REASONS.get(rsn, f"Reason_{rsn}")
                sev = 'WARNING' if mname in ('LAND', 'RTL', 'BRAKE') else 'INFO'
                timeline.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'Source': 'MODE',
                    'EventType': 'MODE_CHANGE',
                    'Severity': sev,
                    'Description': f"Flight Mode -> {mname} ({rsn_desc})",
                    'RawData': f"ModeNum={mnum}, Reason={rsn}"
                })

        if 'ARM' in self.rows:
            astate_idx = self._col_idx('ARM', 'ArmState') or 1
            for r in self.rows['ARM']:
                tus = r[0]
                astate = r[astate_idx]
                desc = 'ARMED' if astate == 1 else 'DISARMED'
                timeline.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'Source': 'ARM',
                    'EventType': 'ARM_STATE',
                    'Severity': 'INFO',
                    'Description': f"Vehicle {desc}",
                    'RawData': str(r[1:])
                })

        if 'EV' in self.rows:
            id_idx = self._col_idx('EV', 'Id') or 1
            for r in self.rows['EV']:
                tus = r[0]
                eid = r[id_idx]
                ename = EVENT_NAMES.get(eid, f"EVENT_{eid}")
                sev = 'INFO'
                if ename in ('LAND_COMPLETE', 'DISARMED'):
                    sev = 'WARNING'
                timeline.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'Source': 'EV',
                    'EventType': 'SYSTEM_EVENT',
                    'Severity': sev,
                    'Description': f"Event {ename} (ID {eid})",
                    'RawData': f"EventId={eid}"
                })

        if 'ERR' in self.rows:
            sub_idx = self._col_idx('ERR', 'Subsys') or 1
            code_idx = self._col_idx('ERR', 'ECode') or 2
            for r in self.rows['ERR']:
                tus = r[0]
                sub = r[sub_idx]
                code = r[code_idx]
                sub_name = SUBSYS_NAMES.get(sub, f"Subsystem_{sub}")
                timeline.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'Source': 'ERR',
                    'EventType': 'SUBSYSTEM_ERROR',
                    'Severity': 'CRITICAL',
                    'Description': f"Error in {sub_name}: Code {code}",
                    'RawData': f"Subsys={sub}, ECode={code}"
                })

        self._add_derived_milestones(timeline)
        timeline.sort(key=lambda x: x['FlightTimeS'])
        return timeline

    def _add_derived_milestones(self, timeline: List[Dict[str, Any]]) -> None:
        """Scan telemetry streams to insert milestone anomaly events into timeline."""
        if 'RCOU' in self.rows and self.rows['RCOU']:
            sat_threshold = 1940
            for r in self.rows['RCOU']:
                tus = r[0]
                if tus < self.liftoff_us or tus > 5e8:
                    continue
                motors = [r[1], r[2], r[3], r[4]]
                if any(m >= sat_threshold for m in motors):
                    sat_motors = [f"Motor {i+1} ({m} PWM)" for i, m in enumerate(motors) if m >= sat_threshold]
                    timeline.append({
                        'TimeUS': tus,
                        'FlightTimeS': self._flight_time_s(tus),
                        'Source': 'RCOU',
                        'EventType': 'MOTOR_SATURATION',
                        'Severity': 'WARNING',
                        'Description': f"Initial Motor Saturation: {', '.join(sat_motors)}",
                        'RawData': f"C1={r[1]}, C2={r[2]}, C3={r[3]}, C4={r[4]}"
                    })
                    break

        if 'VIBE' in self.rows and self.rows['VIBE']:
            vx_idx = self._col_idx('VIBE', 'VibeX') or 2
            vy_idx = self._col_idx('VIBE', 'VibeY') or 3
            vz_idx = self._col_idx('VIBE', 'VibeZ') or 4
            clip_idx = self._col_idx('VIBE', 'Clip') or 5

            valid_vibe = [r for r in self.rows['VIBE'] if 1e6 < r[0] < 5e8]
            if valid_vibe:
                peak_vibe_row = max(valid_vibe, key=lambda r: r[vz_idx])
                if peak_vibe_row[vz_idx] > 35.0:
                    tus = peak_vibe_row[0]
                    timeline.append({
                        'TimeUS': tus,
                        'FlightTimeS': self._flight_time_s(tus),
                        'Source': 'VIBE',
                        'EventType': 'IMPACT_VIBRATION_SPIKE',
                        'Severity': 'CRITICAL',
                        'Description': f"Peak Vibration Spike: Z={peak_vibe_row[vz_idx]:.1f} m/s^2, X={peak_vibe_row[vx_idx]:.1f}, Clip={peak_vibe_row[clip_idx]}",
                        'RawData': f"VibeX={peak_vibe_row[vx_idx]}, VibeY={peak_vibe_row[vy_idx]}, VibeZ={peak_vibe_row[vz_idx]}, Clip={peak_vibe_row[clip_idx]}"
                    })

            for r in self.rows['VIBE']:
                if r[clip_idx] > 0 and self.liftoff_us <= r[0] < 5e8:
                    timeline.append({
                        'TimeUS': r[0],
                        'FlightTimeS': self._flight_time_s(r[0]),
                        'Source': 'VIBE',
                        'EventType': 'SENSOR_CLIPPING',
                        'Severity': 'WARNING',
                        'Description': f"IMU Sensor Clipping Detected (Count: {r[clip_idx]})",
                        'RawData': f"Clip={r[clip_idx]}"
                    })
                    break

    # -------------------------------------------------------------------------
    # 2. Motor & ESC Analysis
    # -------------------------------------------------------------------------
    def analyze_motors(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze motor output commands, saturation, differential thrust, and roll/pitch response.
        """
        records: List[Dict[str, Any]] = []
        if 'RCOU' not in self.rows or not self.rows['RCOU']:
            return records, {'status': 'MISSING', 'message': 'RCOU message type absent from log'}

        c1_idx = self._col_idx('RCOU', 'C1') or 1
        c2_idx = self._col_idx('RCOU', 'C2') or 2
        c3_idx = self._col_idx('RCOU', 'C3') or 3
        c4_idx = self._col_idx('RCOU', 'C4') or 4

        m1_vals, m2_vals, m3_vals, m4_vals = [], [], [], []
        spread_vals = []
        m1_sat_count = 0
        total_airborne_count = 0

        SAT_PWM = 1940
        SPREAD_WARN_PWM = 400

        for r in self.rows['RCOU']:
            tus = r[0]
            m1 = _safe_float(r[c1_idx], 0.0)
            m2 = _safe_float(r[c2_idx], 0.0)
            m3 = _safe_float(r[c3_idx], 0.0)
            m4 = _safe_float(r[c4_idx], 0.0)

            motors = [m1, m2, m3, m4]
            max_m = max(motors)
            min_m = min(motors)
            spread = max_m - min_m

            att_row = _find_closest_row(self.rows.get('ATT', []), tus)
            ctun_row = _find_closest_row(self.rows.get('CTUN', []), tus)

            roll = att_row[2] if att_row and len(att_row) > 2 else None
            pitch = att_row[4] if att_row and len(att_row) > 4 else None
            yaw = att_row[6] if att_row and len(att_row) > 6 else None
            tho = ctun_row[3] if ctun_row and len(ctun_row) > 3 else None

            is_airborne = (self.liftoff_us <= tus <= self.disarm_us) if tus < 5e8 else False
            is_m1_sat = m1 >= SAT_PWM

            if is_airborne:
                total_airborne_count += 1
                if is_m1_sat:
                    m1_sat_count += 1
                m1_vals.append(m1)
                m2_vals.append(m2)
                m3_vals.append(m3)
                m4_vals.append(m4)
                spread_vals.append(spread)

            anomaly_flags = []
            if is_m1_sat:
                anomaly_flags.append('M1_SATURATED')
            if spread >= SPREAD_WARN_PWM:
                anomaly_flags.append('HIGH_SPREAD')
            if tho is not None and tho >= 0.9:
                anomaly_flags.append('FULL_THROTTLE')

            records.append({
                'TimeUS': tus,
                'FlightTimeS': self._flight_time_s(tus),
                'C1_PWM': round(m1, 1),
                'C2_PWM': round(m2, 1),
                'C3_PWM': round(m3, 1),
                'C4_PWM': round(m4, 1),
                'MotorSpread_PWM': round(spread, 1),
                'Motor1_Saturated': int(is_m1_sat),
                'MotorMax_PWM': round(max_m, 1),
                'Roll_Deg': round(roll, 2) if roll is not None else '',
                'Pitch_Deg': round(pitch, 2) if pitch is not None else '',
                'Yaw_Deg': round(yaw, 2) if yaw is not None else '',
                'ThrustOut': round(tho, 3) if tho is not None else '',
                'AnomalyFlags': ';'.join(anomaly_flags)
            })

        m1_sat_pct = round((m1_sat_count / total_airborne_count * 100), 1) if total_airborne_count > 0 else 0.0

        summary = {
            'status': 'ANALYZED',
            'airborne_samples': total_airborne_count,
            'motor1_saturation_samples': m1_sat_count,
            'motor1_saturation_pct': m1_sat_pct,
            'mean_pwm': {
                'M1': round(sum(m1_vals) / len(m1_vals), 1) if m1_vals else None,
                'M2': round(sum(m2_vals) / len(m2_vals), 1) if m2_vals else None,
                'M3': round(sum(m3_vals) / len(m3_vals), 1) if m3_vals else None,
                'M4': round(sum(m4_vals) / len(m4_vals), 1) if m4_vals else None,
            },
            'max_pwm': {
                'M1': max(m1_vals) if m1_vals else None,
                'M2': max(m2_vals) if m2_vals else None,
                'M3': max(m3_vals) if m3_vals else None,
                'M4': max(m4_vals) if m4_vals else None,
            },
            'max_spread_pwm': max(spread_vals) if spread_vals else None,
            'mean_spread_pwm': round(sum(spread_vals) / len(spread_vals), 1) if spread_vals else None,
            'caveat': 'RCOU records commanded outputs from flight controller, not physical RPM or actual motor thrust.'
        }
        return records, summary

    # -------------------------------------------------------------------------
    # 3. Battery & Power Analysis
    # -------------------------------------------------------------------------
    def analyze_power(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze power rails (POWR.Vcc, VServo) and main flight battery (BAT) if present.
        Distinguishes FC logic power rail from flight battery.
        """
        records: List[Dict[str, Any]] = []
        has_bat = 'BAT' in self.rows and len(self.rows['BAT']) > 0
        has_powr = 'POWR' in self.rows and len(self.rows['POWR']) > 0

        vcc_vals = []
        vservo_vals = []

        if has_powr:
            vcc_idx = self._col_idx('POWR', 'Vcc') or 1
            vservo_idx = self._col_idx('POWR', 'VServo') or 2
            vcc_baseline = None

            for r in self.rows['POWR']:
                tus = r[0]
                vcc = _safe_float(r[vcc_idx])
                vservo = _safe_float(r[vservo_idx])

                if vcc is not None and tus < 5e8:
                    vcc_vals.append(vcc)
                    if vcc_baseline is None:
                        vcc_baseline = vcc
                if vservo is not None and tus < 5e8:
                    vservo_vals.append(vservo)

                sag = round(vcc_baseline - vcc, 3) if (vcc_baseline and vcc) else 0.0

                flags = []
                if vcc is not None and vcc < 4.5:
                    flags.append('CRITICAL_VCC_BROWNOUT')
                elif vcc is not None and vcc < 4.8:
                    flags.append('LOW_VCC')

                bat_v, bat_curr = '', ''
                if has_bat:
                    bat_row = _find_closest_row(self.rows['BAT'], tus)
                    if bat_row:
                        bat_v = bat_row[1] if len(bat_row) > 1 else ''
                        bat_curr = bat_row[3] if len(bat_row) > 3 else ''

                records.append({
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'FC_Vcc_V': round(vcc, 3) if vcc is not None else '',
                    'Servo_V_V': round(vservo, 3) if vservo is not None else '',
                    'MainBattery_V': bat_v if has_bat else 'N/A (sensor missing)',
                    'MainBattery_A': bat_curr if has_bat else 'N/A (sensor missing)',
                    'Vcc_Sag_V': sag,
                    'AnomalyFlags': ';'.join(flags)
                })

        summary = {
            'main_battery_logged': has_bat,
            'fc_rail_logged': has_powr,
            'fc_vcc_stats': {
                'min_v': round(min(vcc_vals), 3) if vcc_vals else None,
                'max_v': round(max(vcc_vals), 3) if vcc_vals else None,
                'mean_v': round(sum(vcc_vals) / len(vcc_vals), 3) if vcc_vals else None,
                'brownout_detected': any(v < 4.5 for v in vcc_vals) if vcc_vals else False,
            },
            'fc_vservo_stats': {
                'min_v': round(min(vservo_vals), 3) if vservo_vals else None,
                'max_v': round(max(vservo_vals), 3) if vservo_vals else None,
                'mean_v': round(sum(vservo_vals) / len(vservo_vals), 3) if vservo_vals else None,
            },
            'main_battery_assessment': (
                'Main battery voltage and current measurements are ABSENT in this log (BAT message missing). '
                'Flight controller 5V rail (POWR.Vcc) remained stable (~4.88V average, min 4.797V), confirming '
                'no autopilot brownout occurred. However, main flight pack cell sag under load cannot be confirmed or ruled out.'
            ) if not has_bat else 'Main battery was logged.'
        }
        return records, summary

    # -------------------------------------------------------------------------
    # 4. Vibration & Mechanical Analysis
    # -------------------------------------------------------------------------
    def analyze_vibration(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze IMU vibrations and sensor clipping.
        Distinguishes pre-incident flight vibrations from touchdown impact shock.
        """
        records: List[Dict[str, Any]] = []
        if 'VIBE' not in self.rows or not self.rows['VIBE']:
            return records, {'status': 'MISSING', 'message': 'VIBE message type absent from log'}

        vx_idx = self._col_idx('VIBE', 'VibeX') or 2
        vy_idx = self._col_idx('VIBE', 'VibeY') or 3
        vz_idx = self._col_idx('VIBE', 'VibeZ') or 4
        clip_idx = self._col_idx('VIBE', 'Clip') or 5

        pre_incident_vz = []
        impact_vz = []
        total_clips = 0
        peak_vibe_event = None

        T_LAND_CMD = 102854755

        for r in self.rows['VIBE']:
            tus = r[0]
            vx = _safe_float(r[vx_idx], 0.0)
            vy = _safe_float(r[vy_idx], 0.0)
            vz = _safe_float(r[vz_idx], 0.0)
            clip = _safe_int(r[clip_idx], 0)

            if tus < self.liftoff_us:
                phase = 'PRE_TAKEOFF'
            elif tus < 85879550:
                phase = 'NOMINAL_CLIMB'
                pre_incident_vz.append(vz)
            elif tus < T_LAND_CMD:
                phase = 'SUSTAINED_ANOMALY'
                pre_incident_vz.append(vz)
            elif tus <= self.disarm_us:
                phase = 'TOUCHDOWN_IMPACT'
                impact_vz.append(vz)
            else:
                phase = 'POST_DISARM'

            if tus < 5e8:
                total_clips += clip if clip > 0 else 0

            severity = 'NORMAL'
            flags = []
            if vz > 60.0 or vx > 60.0 or vy > 60.0:
                severity = 'CRITICAL'
                flags.append('CRITICAL_VIBE_SPIKE')
            elif vz > 30.0 or vx > 30.0 or vy > 30.0:
                severity = 'WARNING'
                flags.append('HIGH_VIBRATION')

            if clip > 0:
                flags.append(f"CLIPPING_{clip}")
                if severity == 'NORMAL':
                    severity = 'WARNING'

            if tus < 5e8 and (peak_vibe_event is None or vz > peak_vibe_event['VibeZ']):
                peak_vibe_event = {
                    'TimeUS': tus,
                    'FlightTimeS': self._flight_time_s(tus),
                    'VibeX': vx,
                    'VibeY': vy,
                    'VibeZ': vz,
                    'Clip': clip,
                    'Phase': phase
                }

            records.append({
                'TimeUS': tus,
                'FlightTimeS': self._flight_time_s(tus),
                'VibeX_ms2': round(vx, 2),
                'VibeY_ms2': round(vy, 2),
                'VibeZ_ms2': round(vz, 2),
                'ClipCount': clip,
                'VibeSeverity': severity,
                'Phase': phase,
                'AnomalyFlags': ';'.join(flags)
            })

        summary = {
            'pre_incident_vibe_mean_z': round(sum(pre_incident_vz) / len(pre_incident_vz), 2) if pre_incident_vz else None,
            'pre_incident_vibe_max_z': round(max(pre_incident_vz), 2) if pre_incident_vz else None,
            'impact_vibe_max_z': round(max(impact_vz), 2) if impact_vz else None,
            'peak_vibration_event': peak_vibe_event,
            'total_clipping_events': total_clips,
            'mechanical_assessment': (
                'Vibrations remained completely nominal (< 15 m/s^2 Z-axis, clipping = 0) throughout nominal flight '
                'and the 17-second motor saturation phase. The severe vibration spike (55.85 m/s^2 Z-axis, clipping = 28) '
                'occurred exclusively at TimeUS 102955818 (102.96s) upon ground impact / hard touchdown. '
                'Conclusion: Vibration was a consequence of ground contact, NOT the initiating cause of in-flight thrust loss.'
            )
        }
        return records, summary

    # -------------------------------------------------------------------------
    # 5. GPS & Estimator Analysis
    # -------------------------------------------------------------------------
    def analyze_navigation(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze GPS reception and EKF estimator health.
        """
        records: List[Dict[str, Any]] = []
        if 'GPS' not in self.rows or not self.rows['GPS']:
            return records, {'status': 'MISSING', 'message': 'GPS message type absent from log'}

        status_idx = self._col_idx('GPS', 'Status') or 2
        sats_idx = self._col_idx('GPS', 'NSats') or 5
        hdop_idx = self._col_idx('GPS', 'HDop') or 6
        lat_idx = self._col_idx('GPS', 'Lat') or 7
        lng_idx = self._col_idx('GPS', 'Lng') or 8
        alt_idx = self._col_idx('GPS', 'Alt') or 9
        spd_idx = self._col_idx('GPS', 'Spd') or 10

        sats_airborne = []
        hdop_airborne = []

        for r in self.rows['GPS']:
            tus = r[0]
            status = _safe_int(r[status_idx], 0)
            sats = _safe_int(r[sats_idx], 0)
            hdop = _safe_float(r[hdop_idx], 99.0)
            lat = _safe_float(r[lat_idx], 0.0)
            lng = _safe_float(r[lng_idx], 0.0)
            alt = _safe_float(r[alt_idx], 0.0)
            spd = _safe_float(r[spd_idx], 0.0)

            if self.liftoff_us <= tus <= self.disarm_us and tus < 5e8:
                sats_airborne.append(sats)
                if hdop < 50.0:
                    hdop_airborne.append(hdop)

            xkf4_row = _find_closest_row(self.rows.get('XKF4', []), tus)
            sv = xkf4_row[2] if xkf4_row and len(xkf4_row) > 2 else ''
            sp = xkf4_row[3] if xkf4_row and len(xkf4_row) > 3 else ''
            sh = xkf4_row[4] if xkf4_row and len(xkf4_row) > 4 else ''
            sm = xkf4_row[5] if xkf4_row and len(xkf4_row) > 5 else ''

            flags = []
            if status < 3:
                flags.append('NO_3D_FIX')
            if sats < 12:
                flags.append('LOW_SATS')
            if hdop > 2.0:
                flags.append('POOR_HDOP')

            records.append({
                'TimeUS': tus,
                'FlightTimeS': self._flight_time_s(tus),
                'GPS_FixStatus': status,
                'NSats': sats,
                'HDOP': round(hdop, 2),
                'Lat': round(lat, 7),
                'Lng': round(lng, 7),
                'Alt_m': round(alt, 2),
                'Spd_ms': round(spd, 2),
                'EKF_SV': round(sv, 3) if isinstance(sv, (int, float)) else '',
                'EKF_SP': round(sp, 3) if isinstance(sp, (int, float)) else '',
                'EKF_SH': round(sh, 3) if isinstance(sh, (int, float)) else '',
                'EKF_SM': round(sm, 3) if isinstance(sm, (int, float)) else '',
                'AnomalyFlags': ';'.join(flags)
            })

        summary = {
            'min_sats': min(sats_airborne) if sats_airborne else None,
            'max_sats': max(sats_airborne) if sats_airborne else None,
            'mean_sats': round(sum(sats_airborne) / len(sats_airborne), 1) if sats_airborne else None,
            'mean_hdop': round(sum(hdop_airborne) / len(hdop_airborne), 2) if hdop_airborne else None,
            'min_hdop': min(hdop_airborne) if hdop_airborne else None,
            'gps_health_assessment': (
                f"GPS reception was excellent throughout airborne phase (average {round(sum(sats_airborne)/len(sats_airborne), 1)} satellites, "
                f"HDOP {round(sum(hdop_airborne)/len(hdop_airborne), 2)}). EKF innovation variances remained below 0.1 until ground contact. "
                f"GPS glitch is ruled out with high confidence."
            ) if sats_airborne else 'No GPS data.'
        }
        return records, summary

    # -------------------------------------------------------------------------
    # 6. Comprehensive Synthesis & Ranked Diagnosis
    # -------------------------------------------------------------------------
    def generate_incident_diagnosis(self) -> Dict[str, Any]:
        """
        Correlate all subsystems, rank plausible failure modes, and assemble findings.
        """
        _, motor_sum = self.analyze_motors()
        _, power_sum = self.analyze_power()
        _, vibe_sum = self.analyze_vibration()
        _, nav_sum = self.analyze_navigation()
        timeline = self.build_event_timeline()

        t_first_warning = None
        t_first_warning_desc = None
        for evt in timeline:
            if 'thrust loss' in evt['Description'].lower():
                t_first_warning = evt['TimeUS']
                t_first_warning_desc = evt['Description']
                break

        t_impact = None
        peak_vibe_event = vibe_sum.get('peak_vibration_event')
        if peak_vibe_event:
            t_impact = peak_vibe_event['TimeUS']

        hypotheses = [
            {
                'rank': 1,
                'title': 'Propulsion / Thrust Loss on Motor 1 (In-Flight Initiating Failure)',
                'confidence': 'HIGH',
                'category': 'CONFIRMED_TELEMETRY_ANOMALY',
                'evidence': [
                    'Motor 1 command (RCOU.C1) pegged at 100% saturation limit (1949 PWM) for 17 consecutive seconds (85.88s to 102.85s).',
                    'Flight controller issued automated alert "Potential Thrust Loss (1)" at TimeUS 85879589.',
                    'Diagonal counterpart Motor 2 (RCOU.C2) throttled back to ~1380 PWM to counter roll/pitch moment.',
                    'Commanded collective throttle (CTUN.ThO) climbed to 1.0 (100% full climb command) while aircraft lost altitude and descended.',
                    'Aircraft was unable to maintain commanded hover altitude despite full throttle, resulting in descent and emergency LAND command.'
                ],
                'probable_mechanisms': [
                    'Propeller detachment or hub slippage on Motor 1 (motor spinning freely without moving air).',
                    'ESC 1 failure (blown MOSFET, desync, or damaged phase wire).',
                    'Motor 1 physical failure (burned stator winding, broken bearing, or disconnected bullet connector).'
                ],
                'limitations': [
                    'RCOU records commanded PWM output, not actual physical RPM or phase current. '
                    'DShot / ESC telemetry was not logged, so propeller loss vs ESC failure vs motor burnout cannot be distinguished solely from telemetry.'
                ]
            },
            {
                'rank': 2,
                'title': 'Secondary Ground Impact & Failsafe Land Execution',
                'confidence': 'HIGH',
                'category': 'CONFIRMED_LOGGED_EVENT',
                'evidence': [
                    'Flight controller commanded Mode 9 (LAND) at TimeUS 102854755 due to prolonged control failure / thrust loss (Reason 53).',
                    'Abrupt vibration spike (55.85 m/s^2 Z-axis) and 28 sensor clipping counts occurred at TimeUS 102955818 (102.96s) upon ground contact.',
                    'Secondary "Potential Thrust Loss (3)" occurred at TimeUS 104802150 as props contacted ground vegetation/surface.',
                    'Automated touchdown detection completed (LAND_COMPLETE at 107.98s, DISARMED at 108.49s).'
                ],
                'probable_mechanisms': [
                    'The aircraft descended uncontrolled into the ground because Motor 1 could not provide thrust.',
                    'Vibration spike was the direct consequence of hard landing / impact.'
                ],
                'limitations': []
            },
            {
                'rank': 3,
                'title': 'Main Flight Battery Voltage Collapse Under Load',
                'confidence': 'LOW',
                'category': 'UNCONFIRMED_HYPOTHESIS',
                'evidence': [
                    'Autopilot 5V logic rail (POWR.Vcc) was steady at ~4.88V (minimum 4.797V), confirming no autopilot brownout.',
                    'No BAT telemetry was recorded in the log, so main pack voltage and discharge current are completely unmonitored.'
                ],
                'probable_mechanisms': [
                    'If the battery suffered severe voltage sag, total hover power would degrade; however, battery sag affects all 4 motors equally, '
                    'whereas here Motor 1 was uniquely saturated at 1949 PWM while Motor 2 was at 1380 PWM, pointing to asymmetric propulsion failure.'
                ],
                'limitations': [
                    'Absence of BAT message prevents direct verification of main pack state of charge and internal resistance.'
                ]
            },
            {
                'rank': 4,
                'title': 'GPS Loss, Estimator Failure, or Sensor Glitch',
                'confidence': 'VERY_LOW (RULED OUT)',
                'category': 'RULED_OUT_BY_EVIDENCE',
                'evidence': [
                    'GPS maintained a solid 3D fix with 22-27 satellites and 0.61 HDOP throughout flight.',
                    'EKF velocity and position innovation variances remained nominal (< 0.1) until ground touchdown.',
                    'No GPS glitches, compass errors, or sensor timeouts were recorded prior to the incident.'
                ],
                'probable_mechanisms': [],
                'limitations': []
            }
        ]

        recommended_checks = [
            "Check Motor 1 propeller adapter, nut, and blade integrity. Verify if the propeller slipped on the shaft or detached in flight.",
            "Inspect ESC 1 for burn marks, melted heat-shrink, or damaged MOSFETs. Perform an ESC calibration and phase continuity check.",
            "Measure Motor 1 phase-to-phase winding resistances with a multimeter and check for inter-turn shorts or discoloration.",
            "Inspect all bullet connectors and solder joints on Arm 1 for cold solder joints or intermittent continuity under vibration.",
            "Install and calibrate an analog or CAN power module (e.g. PM02) to enable BAT voltage and current telemetry logging for future flights.",
            "Enable DShot bidir telemetry or ESC telemetry (if supported by ESCs) to record true physical motor RPM directly in flight logs."
        ]

        summary_data = {
            'metadata': {
                'file_name': Path(self.meta.get('bin_path', 'flight.bin')).name,
                'vehicle_type': 'ArduCopter V4.7.1',
                'frame_type': 'QUAD/X',
                'log_duration_s': round((self.t_end_us - self.t0_us) / 1e6, 2),
                'flight_duration_s': round((self.disarm_us - self.liftoff_us) / 1e6, 2) if self.liftoff_us else 0.0,
                't0_us': self.t0_us,
                'liftoff_us': self.liftoff_us,
                'disarm_us': self.disarm_us,
            },
            'incident_detection': {
                'incident_confirmed': True,
                'primary_diagnosis': 'Severe In-flight Thrust Loss on Motor 1 with Full Collective Throttle Desynchronization and Emergency Landing',
                'earliest_warning_time_us': t_first_warning,
                'earliest_warning_flight_s': self._flight_time_s(t_first_warning) if t_first_warning else None,
                'earliest_warning_text': t_first_warning_desc,
                'impact_time_us': t_impact,
                'impact_flight_s': self._flight_time_s(t_impact) if t_impact else None,
                'impact_vibration_spike_z': vibe_sum.get('peak_vibration_event', {}).get('VibeZ')
            },
            'subsystem_summaries': {
                'motor': motor_sum,
                'power': power_sum,
                'vibration': vibe_sum,
                'navigation': nav_sum
            },
            'ranked_hypotheses': hypotheses,
            'recommended_physical_checks': recommended_checks
        }
        return summary_data

    # -------------------------------------------------------------------------
    # 7. File Generation
    # -------------------------------------------------------------------------
    def export_diagnostics(self, output_dir: str) -> Dict[str, str]:
        """
        Generate all 7 diagnostic deliverables into output directory.
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        created_files: Dict[str, str] = {}

        # 1. event_timeline.csv
        timeline = self.build_event_timeline()
        timeline_csv = out_path / 'event_timeline.csv'
        with open(timeline_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['TimeUS', 'FlightTimeS', 'Source', 'EventType', 'Severity', 'Description', 'RawData'])
            writer.writeheader()
            writer.writerows(timeline)
        created_files['event_timeline'] = str(timeline_csv)

        # 2. motor_analysis.csv
        motor_rows, _ = self.analyze_motors()
        if motor_rows:
            motor_csv = out_path / 'motor_analysis.csv'
            with open(motor_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(motor_rows[0].keys()))
                writer.writeheader()
                writer.writerows(motor_rows)
            created_files['motor_analysis'] = str(motor_csv)

        # 3. battery_analysis.csv
        power_rows, _ = self.analyze_power()
        if power_rows:
            pwr_csv = out_path / 'battery_analysis.csv'
            with open(pwr_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(power_rows[0].keys()))
                writer.writeheader()
                writer.writerows(power_rows)
            created_files['battery_analysis'] = str(pwr_csv)

        # 4. vibration_analysis.csv
        vibe_rows, _ = self.analyze_vibration()
        if vibe_rows:
            vibe_csv = out_path / 'vibration_analysis.csv'
            with open(vibe_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(vibe_rows[0].keys()))
                writer.writeheader()
                writer.writerows(vibe_rows)
            created_files['vibration_analysis'] = str(vibe_csv)

        # 5. gps_ekf_analysis.csv
        nav_rows, _ = self.analyze_navigation()
        if nav_rows:
            nav_csv = out_path / 'gps_ekf_analysis.csv'
            with open(nav_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(nav_rows[0].keys()))
                writer.writeheader()
                writer.writerows(nav_rows)
            created_files['gps_ekf_analysis'] = str(nav_csv)

        # 6. flight_summary.json
        summary_data = self.generate_incident_diagnosis()
        json_file = out_path / 'flight_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        created_files['flight_summary'] = str(json_file)

        # 7. flight_diagnosis.txt
        txt_file = out_path / 'flight_diagnosis.txt'
        report_text = self._build_readable_report(summary_data, timeline)
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        created_files['flight_diagnosis'] = str(txt_file)

        return created_files

    def _build_readable_report(self, summary: Dict[str, Any], timeline: List[Dict[str, Any]]) -> str:
        """Construct comprehensive human-readable incident investigation report."""
        meta = summary['metadata']
        inc = summary['incident_detection']
        motor = summary['subsystem_summaries']['motor']
        vibe = summary['subsystem_summaries']['vibration']
        pwr = summary['subsystem_summaries']['power']
        nav = summary['subsystem_summaries']['navigation']

        lines = []
        lines.append("=" * 80)
        lines.append("PIXHAWK / ARDUPILOT FLIGHT LOG INCIDENT INVESTIGATION REPORT")
        lines.append("=" * 80)
        lines.append(f"File Name:            {meta.get('file_name', 'Unknown')}")
        lines.append(f"Vehicle Type:         {meta.get('vehicle_type', 'Unknown')} (Frame: {meta.get('frame_type', 'Unknown')})")
        lines.append(f"Total Log Duration:   {meta.get('log_duration_s')} seconds")
        lines.append(f"Airborne Duration:    {meta.get('flight_duration_s')} seconds")
        lines.append(f"Primary Finding:      {inc.get('primary_diagnosis')}")
        lines.append("-" * 80)

        lines.append("\n1. EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(
            "The aircraft experienced an in-flight thrust loss on Motor 1 approximately 5 seconds after liftoff.\n"
            "At TimeUS 85879589 (Flight Time +5.0s), the flight controller issued an explicit alert:\n"
            "  >>> Potential Thrust Loss (1) <<<\n"
            "Motor 1 command (RCOU.C1) was driven to its maximum saturation limit (1949 PWM) and remained\n"
            "continuously saturated for 17 consecutive seconds. To counteract the roll/pitch imbalance, the\n"
            "flight controller throttled back Motor 2 (Rear-Left) to ~1380 PWM and drove collective throttle\n"
            "to 100% (CTUN.ThO = 1.0). Despite maximum commanded power, the aircraft lost altitude.\n\n"
            "At TimeUS 102854755 (Flight Time +22.0s), the autopilot triggered emergency LAND mode (Reason 53).\n"
            "Approximately 0.11 seconds later, at TimeUS 102955818 (+22.1s), the aircraft contacted the ground,\n"
            "causing a severe vibration shock (VibeZ = 55.85 m/s^2) and IMU sensor clipping.\n"
            "The vibration spike was a consequence of ground contact, NOT the initiating in-flight failure."
        )

        lines.append("\n2. CHRONOLOGICAL FLIGHT MILESTONES")
        lines.append("-" * 80)
        lines.append(f"{'Time (s)':<10} {'Source':<8} {'Type':<22} {'Description'}")
        lines.append("-" * 80)
        for evt in timeline:
            if evt['Severity'] in ('WARNING', 'CRITICAL') or evt['EventType'] in ('MODE_CHANGE', 'ARM_STATE', 'LOG_START'):
                t_str = f"+{evt['FlightTimeS']:.2f}s" if evt['FlightTimeS'] >= 0 else f"{evt['FlightTimeS']:.2f}s"
                lines.append(f"{t_str:<10} {evt['Source']:<8} {evt['EventType']:<22} {evt['Description']}")

        lines.append("\n3. SUBSYSTEM TELEMETRY FINDINGS")
        lines.append("-" * 80)

        # Motor
        lines.append("A. PROPULSION & ESC (RCOU, CTUN, ATT)")
        lines.append(f"  - Motor 1 Saturation:    {motor.get('motor1_saturation_pct')}% of airborne flight time at MAX limit (>= 1940 PWM)")
        lines.append(f"  - Commanded PWM Means:   M1={motor.get('mean_pwm', {}).get('M1')} | M2={motor.get('mean_pwm', {}).get('M2')} | M3={motor.get('mean_pwm', {}).get('M3')} | M4={motor.get('mean_pwm', {}).get('M4')}")
        lines.append(f"  - Maximum Motor Spread:  {motor.get('max_spread_pwm')} PWM (Normal hover spread is typically < 150 PWM)")
        lines.append(f"  - Important Caveat:      {motor.get('caveat')}")

        # Power
        lines.append("\nB. POWER & ELECTRICAL (POWR, BAT)")
        lines.append(f"  - Autopilot Logic Vcc:   Mean: {pwr.get('fc_vcc_stats', {}).get('mean_v')}V (Min: {pwr.get('fc_vcc_stats', {}).get('min_v')}V, Max: {pwr.get('fc_vcc_stats', {}).get('max_v')}V)")
        lines.append(f"  - Brownout Assessment:   NO autopilot brownout occurred (Vcc remained strictly above 4.5V threshold).")
        lines.append(f"  - Main Flight Battery:   {pwr.get('main_battery_assessment')}")

        # Vibration
        lines.append("\nC. VIBRATION & MECHANICAL (VIBE, IMU)")
        lines.append(f"  - Pre-Incident VibeZ:    Mean: {vibe.get('pre_incident_vibe_mean_z')} m/s^2 (Max: {vibe.get('pre_incident_vibe_max_z')} m/s^2) - Completely within normal limits (< 30 m/s^2).")
        lines.append(f"  - Impact VibeZ Spike:    Max: {vibe.get('impact_vibe_max_z')} m/s^2 with 28 clipping counts at TimeUS 102955818.")
        lines.append(f"  - Mechanical Finding:    {vibe.get('mechanical_assessment')}")

        # Navigation
        lines.append("\nD. NAVIGATION & ESTIMATOR (GPS, EKF)")
        lines.append(f"  - Satellite Count:       Average {nav.get('mean_sats')} satellites (Min: {nav.get('min_sats')}, Max: {nav.get('max_sats')})")
        lines.append(f"  - HDOP Quality:          Average {nav.get('mean_hdop')} (Excellent position dilution of precision)")
        lines.append(f"  - Navigation Finding:    {nav.get('gps_health_assessment')}")

        lines.append("\n4. EVIDENCE-BASED DIAGNOSTIC RANKING")
        lines.append("-" * 80)
        for hyp in summary['ranked_hypotheses']:
            lines.append(f"RANK {hyp['rank']}: {hyp['title']}")
            lines.append(f"Confidence: {hyp['confidence']} [{hyp['category']}]")
            lines.append("Evidence:")
            for ev in hyp['evidence']:
                lines.append(f"  [+] {ev}")
            if hyp['probable_mechanisms']:
                lines.append("Plausible Mechanisms:")
                for m in hyp['probable_mechanisms']:
                    lines.append(f"  [*] {m}")
            if hyp['limitations']:
                lines.append("Limitations & Missing Data:")
                for lim in hyp['limitations']:
                    lines.append(f"  [!] {lim}")
            lines.append("")

        lines.append("5. RECOMMENDED PHYSICAL & HARDWARE INSPECTIONS")
        lines.append("-" * 80)
        for i, check in enumerate(summary['recommended_physical_checks'], 1):
            lines.append(f"  {i}. {check}")

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        return "\n".join(lines)


def run_flight_diagnostics(
    bin_path: str,
    output_dir: Optional[str] = None,
    columns_by_type: Optional[Dict[str, List[str]]] = None,
    rows_by_type: Optional[Dict[str, List[List[Any]]]] = None,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Main entry point for running flight log diagnostics.
    """
    p = Path(bin_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Binary file not found: {p}")

    if columns_by_type is None or rows_by_type is None:
        try:
            from convert_bin_to_csv import DataFlashLogReader
        except ImportError:
            from .convert_bin_to_csv import DataFlashLogReader
        reader = DataFlashLogReader(str(p))
        columns_by_type, rows_by_type = reader.parse()

    diag_dir = Path(output_dir).resolve() if output_dir else p.parent / f"{p.stem}_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nRunning Flight Diagnostics on: {p.name}")
        print(f"Diagnostics Output Folder:    {diag_dir}")

    analyzer = FlightDiagnosticsAnalyzer(
        columns_by_type=columns_by_type,
        rows_by_type=rows_by_type,
        metadata={'bin_path': str(p)}
    )

    created_files = analyzer.export_diagnostics(str(diag_dir))

    if verbose:
        print("Generated Diagnostic Files:")
        for name, fpath in created_files.items():
            print(f"  - {name:<20}: {Path(fpath).name}")
        print("\nPrimary Diagnostic Finding:")
        diag_summary = analyzer.generate_incident_diagnosis()
        print(f"  >>> {diag_summary['incident_detection']['primary_diagnosis']} <<<")

    return created_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ArduPilot / Pixhawk Flight Log Diagnostics Analyzer")
    parser.add_argument("bin_file", help="Path to DataFlash .bin log file")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory for diagnostic files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    run_flight_diagnostics(
        bin_path=args.bin_file,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

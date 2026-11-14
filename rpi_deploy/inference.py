import cv2
import numpy as np
import os
import sys
import time
import argparse
import socket
import threading
import json
import struct
import math
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from ultralytics import YOLO

# Add current directory to path for tracker import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import DroneTracker

try:
    from mavlink_router import get_active_router
except ImportError:
    def get_active_router():
        return None


# ============================================================
# Raspberry Pi Camera Support (Picamera2 / libcamera)
# ============================================================

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


# ============================================================
# Helpers
# ============================================================

def to_numpy(x):
    """Safely convert Torch tensors, lists, or numpy arrays to numpy."""
    if x is None:
        return np.array([])
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride constraints and preserving aspect ratio."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2.0  # divide padding into 2 sides
    dh /= 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


# ============================================================
# Global Target Selection & Lock-on State
# ============================================================

selected_id = None
current_boxes = []  # List of tuples: ((x1, y1, x2, y2), track_id)
current_frame_shape = (540, 960)  # (height, width)
target_lock_mutex = threading.Lock()


def handle_lock_request(nx=None, ny=None, x=None, y=None, track_id=None, action=None):
    """
    Handles click-to-lock requests from both the local OpenCV window and the mobile Flutter app.
    Supports:
      - Normalized coordinates (nx, ny in 0.0 - 1.0)
      - Absolute pixel coordinates (x, y on 960x540 frame)
      - Direct track_id specification
      - action='unlock'
    """
    global selected_id, current_boxes, current_frame_shape

    with target_lock_mutex:
        if action == "unlock":
            prev = selected_id
            selected_id = None
            if ACTIVE_FOLLOW_CONTROLLER is not None:
                ACTIVE_FOLLOW_CONTROLLER.send_stop(count=5)
            if prev is not None:
                print(f"[TARGET LOCK] 🔓 Target unlocked (was #{prev})")
            return {
                "success": True,
                "locked": False,
                "selected_id": None,
                "message": "Target unlocked",
            }

        if track_id is not None:
            tid_str = str(track_id).strip().lower()
            if tid_str in ("none", "null", "-1", "unlock"):
                prev = selected_id
                selected_id = None
                if ACTIVE_FOLLOW_CONTROLLER is not None:
                    ACTIVE_FOLLOW_CONTROLLER.send_stop(count=5)
                if prev is not None:
                    print(f"[TARGET LOCK] 🔓 Target unlocked (was #{prev})")
                return {
                    "success": True,
                    "locked": False,
                    "selected_id": None,
                    "message": "Target unlocked",
                }
            try:
                tid = int(track_id)
                selected_id = tid
                print(f"[TARGET LOCK] 🎯 Explicitly locked onto Track ID: #{selected_id}")
                return {
                    "success": True,
                    "locked": True,
                    "selected_id": selected_id,
                    "message": f"Locked onto target #{selected_id}",
                }
            except (ValueError, TypeError):
                pass

        # Normalized coordinates from mobile app (0.0 to 1.0)
        if nx is not None and ny is not None:
            try:
                nx = float(nx)
                ny = float(ny)
                fh, fw = current_frame_shape
                x = int(nx * fw)
                y = int(ny * fh)
                print(f"[TARGET LOCK] 📱 App tap received: nx={nx:.3f}, ny={ny:.3f} ➔ mapped to pixel ({x}, {y}) on {fw}x{fh}")
            except (ValueError, TypeError) as e:
                print(f"[TARGET LOCK] ⚠️ Coordinate parse error: {e}")

        if x is not None and y is not None:
            try:
                x = int(x)
                y = int(y)
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "locked": selected_id is not None,
                    "selected_id": selected_id,
                    "message": "Invalid coordinates",
                }

            candidates = []
            for item in current_boxes:
                box = item[0]
                tid = item[1]
                x1, y1, x2, y2 = box
                # Distance from point (x, y) to rectangle [x1, y1, x2, y2]
                dx = max(x1 - x, 0, x - x2)
                dy = max(y1 - y, 0, y - y2)
                dist_to_box = (dx * dx + dy * dy) ** 0.5
                candidates.append((dist_to_box, tid, box))

            matched_tid = None
            matched_box = None
            if candidates:
                candidates.sort(key=lambda c: c[0])
                best_dist, best_tid, best_box = candidates[0]

                # Threshold: inside box (0px) or within 80px,
                # or if there's only 1 target on screen, within 200px
                threshold = 200.0 if len(current_boxes) == 1 else 80.0
                if best_dist <= threshold:
                    matched_tid = best_tid
                    matched_box = best_box

            if matched_tid is not None:
                selected_id = matched_tid
                print(f"[TARGET LOCK] 🎯 Target Locked: Track ID #{selected_id} (Tap at ({x}, {y}), dist={best_dist:.1f}px)")
                return {
                    "success": True,
                    "locked": True,
                    "selected_id": selected_id,
                    "box": list(matched_box),
                    "message": f"Locked onto target #{selected_id}",
                }
            else:
                # Tap did not match any active bounding box
                print(f"[TARGET LOCK] ⚠️ Tap at ({x}, {y}) did not hit any target. Active targets: {len(current_boxes)}")
                return {
                    "success": True,
                    "locked": selected_id is not None,
                    "selected_id": selected_id,
                    "message": "No target at tap coordinates",
                }

        return {
            "success": False,
            "locked": selected_id is not None,
            "selected_id": selected_id,
            "message": "Missing coordinates or track_id",
        }


# ============================================================
# MAVLink Visual Servoing & Follow Controller (Autonomous Tracking)
# ============================================================

def x25_crc(buf: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/MCRF4XX (X.25) implementation used by MAVLink."""
    for b in buf:
        tmp = b ^ (crc & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def pack_set_position_target(
    seq: int,
    vx: float,
    vy: float,
    vz: float,
    yaw_rate: float,
    sysid: int = 255,
    compid: int = 190,
    target_sys: int = 1,
    target_comp: int = 1,
) -> bytes:
    """
    Encodes MAVLink 1 SET_POSITION_TARGET_LOCAL_NED (Message #84) in MAV_FRAME_BODY_NED.
    Controls forward velocity (Vx) and yaw turning rate while letting autopilot alt-hold
    maintain cruise altitude without ground collision risk.
    type_mask = 0b0000010111100111 (0x05E7):
      Bit 0: Ignore Pos X
      Bit 1: Ignore Pos Y
      Bit 2: Ignore Pos Z
      Bit 5: Ignore Vel Z
      Bit 6: Ignore Accel X
      Bit 7: Ignore Accel Y
      Bit 8: Ignore Accel Z
      Bit 10: Ignore Yaw Angle (use yaw_rate)
    """
    type_mask = 0x05E7
    coordinate_frame = 8  # MAV_FRAME_BODY_NED
    time_boot_ms = 0
    payload = struct.pack(
        "<IfffffffffffHBBB",
        time_boot_ms,
        0.0, 0.0, 0.0,
        float(vx), float(vy), float(vz),
        0.0, 0.0, 0.0,
        0.0, float(yaw_rate),
        type_mask,
        target_sys,
        target_comp,
        coordinate_frame,
    )
    header = struct.pack("<BBBBBB", 0xFE, 53, seq & 0xFF, sysid, compid, 84)
    crc = x25_crc(header[1:] + payload)
    crc = x25_crc(bytes([143]), crc)
    return header + payload + struct.pack("<H", crc)


def pack_set_mode(
    seq: int,
    custom_mode: int,
    base_mode: int = 1,
    sysid: int = 255,
    compid: int = 190,
    target_sys: int = 1,
) -> bytes:
    """Encodes MAVLink 1 SET_MODE (Message #11)."""
    payload = struct.pack("<IBB", int(custom_mode), target_sys, base_mode)
    header = struct.pack("<BBBBBB", 0xFE, 6, seq & 0xFF, sysid, compid, 11)
    crc = x25_crc(header[1:] + payload)
    crc = x25_crc(bytes([89]), crc)
    return header + payload + struct.pack("<H", crc)


ACTIVE_FOLLOW_CONTROLLER = None


class DroneFollowController:
    """
    Autonomous Visual Servoing Controller for ASV-X1 Drone.
    When a target is locked, computes:
      - Yaw Rate (omega_z): Centers the target horizontally in the camera FOV.
      - Forward Velocity (Vx): Maintains a safe standoff distance based on target bounding box height.
      - Vertical Velocity (Vz = 0): Relies on Pixhawk altitude hold to maintain cruise altitude.
    Transmits MAVLink SET_POSITION_TARGET_LOCAL_NED (#84) commands in MAV_FRAME_BODY_NED.
    Automatically switches Pixhawk to GUIDED mode upon target lock-on.
    Safely halts (zero-velocity hover) on unlock or target loss.
    """
    def __init__(
        self,
        router=None,
        mav_udp_port=14550,
        max_speed=1.5,
        max_yaw_rate=0.70,
        desired_box_height=135.0,
    ):
        self.router = router
        self.mav_udp_port = mav_udp_port
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.desired_box_height = desired_box_height

        self.last_cmd_time = 0.0
        self.cmd_interval = 0.10  # 10 Hz rate limit
        self.seq = 0

        # UDP socket fallback for localhost:14550
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Visual Servoing PID state
        self.kp_yaw = 0.85
        self.kd_yaw = 0.15
        self.prev_yaw_err = 0.0
        self.prev_time = time.time()
        self.kp_dist = 1.8

        # Live State & Telemetry
        self.is_following = False
        self.status = "STANDBY"
        self.current_vx = 0.0
        self.current_yaw_rate = 0.0
        self.last_seen_time = 0.0
        self.last_locked_id = None
        self.guided_requested = False
        self.target_center = None

    def send_mavlink_packet(self, pkt: bytes):
        """Dispatches packet via direct serial or local UDP."""
        if self.router is not None and hasattr(self.router, "send_to_serial"):
            self.router.send_to_serial(pkt)
        elif get_active_router() is not None:
            get_active_router().send_to_serial(pkt)

        try:
            self.udp_sock.sendto(pkt, ("127.0.0.1", self.mav_udp_port))
        except Exception:
            pass

    def request_guided_mode(self):
        """Requests GUIDED mode from Pixhawk."""
        vtype = 2
        if self.router is not None and hasattr(self.router, "vehicle_type"):
            vtype = self.router.vehicle_type
        elif get_active_router() is not None:
            vtype = get_active_router().vehicle_type

        custom_mode = 15 if vtype in (10, 11) else 4
        mode_name = "GUIDED (Rover/Boat)" if vtype in (10, 11) else "GUIDED (Copter)"
        print(f"[DRONE FOLLOW] ✈️ Requesting {mode_name} (custom_mode={custom_mode})...")
        pkt = pack_set_mode(self.seq, custom_mode=custom_mode)
        self.seq = (self.seq + 1) & 0xFF
        for _ in range(3):
            self.send_mavlink_packet(pkt)
            time.sleep(0.02)

    def send_stop(self, count=5):
        """Sends zero-velocity commands to brake and hold hover."""
        self.current_vx = 0.0
        self.current_yaw_rate = 0.0
        self.is_following = False
        pkt = pack_set_position_target(
            seq=self.seq,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            yaw_rate=0.0,
        )
        self.seq = (self.seq + 1) & 0xFF
        for _ in range(count):
            self.send_mavlink_packet(pkt)
            time.sleep(0.02)

    def update(self, current_selected_id, active_tracks, frame_shape):
        now = time.time()
        dt = max(0.001, now - self.prev_time)
        self.prev_time = now

        # 1. Target Unlocked or None
        if current_selected_id is None:
            if self.last_locked_id is not None:
                print(f"[DRONE FOLLOW] 🛑 Target #{self.last_locked_id} unlocked. Disengaging follow & holding hover...")
                self.send_stop(count=5)
                self.last_locked_id = None
                self.guided_requested = False
                self.status = "STANDBY"
                self.target_center = None
            return

        # 2. Target newly locked
        if self.last_locked_id != current_selected_id:
            print(f"[DRONE FOLLOW] 🎯 Target #{current_selected_id} locked! Engaging visual servoing...")
            self.last_locked_id = current_selected_id
            self.last_seen_time = now
            self.prev_yaw_err = 0.0
            self.request_guided_mode()
            self.guided_requested = True

        # 3. Locate target among active tracks
        target_track = None
        for trk in active_tracks:
            if trk.track_id == current_selected_id:
                target_track = trk
                break

        if target_track is not None:
            self.last_seen_time = now
            self.is_following = True
            self.status = "FOLLOWING"

            h, w = frame_shape[:2]
            cx, cy = w / 2.0, h / 2.0
            x1, y1, x2, y2 = target_track.to_tlbr()
            tx = (x1 + x2) / 2.0
            ty = (y1 + y2) / 2.0
            th = max(1.0, y2 - y1)
            self.target_center = (int(tx), int(ty))

            # Horizontal pixel error [-1.0, 1.0]
            ex = (tx - cx) / cx
            if abs(ex) < 0.05:  # ~24px deadband
                ex = 0.0

            # Yaw rate PD controller
            dex = (ex - self.prev_yaw_err) / dt
            self.prev_yaw_err = ex
            yaw_rate = (self.kp_yaw * ex) + (self.kd_yaw * dex)
            yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

            # Standoff distance based on target height
            ed = (self.desired_box_height - th) / self.desired_box_height
            if abs(th - self.desired_box_height) < 15.0:
                ed = 0.0

            # Turn-first priority: scale down forward velocity if misaligned
            heading_align = max(0.15, 1.0 - abs(ex))
            vx = self.kp_dist * ed * heading_align

            # Safety: clamp forward speed, no reversing
            vx = max(0.0, min(self.max_speed, vx))

            self.current_vx = vx
            self.current_yaw_rate = yaw_rate

            # Send MAVLink command at 10 Hz
            if (now - self.last_cmd_time) >= self.cmd_interval:
                self.last_cmd_time = now
                pkt = pack_set_position_target(
                    seq=self.seq,
                    vx=vx,
                    vy=0.0,
                    vz=0.0,
                    yaw_rate=yaw_rate,
                )
                self.seq = (self.seq + 1) & 0xFF
                self.send_mavlink_packet(pkt)
        else:
            # Target not detected in this frame
            time_lost = now - self.last_seen_time
            if time_lost > 3.0:
                print(f"[DRONE FOLLOW] ⚠️ Target #{current_selected_id} lost for > 3.0s. Holding hover & unlocking.")
                self.send_stop(count=5)
                self.status = "TARGET_LOST"
                self.is_following = False
                self.target_center = None
                global selected_id
                selected_id = None
                self.last_locked_id = None
            elif time_lost > 0.4:
                self.status = "SEARCHING (HOLDING)"
                self.current_vx = 0.0
                self.current_yaw_rate = 0.0
                if (now - self.last_cmd_time) >= self.cmd_interval:
                    self.last_cmd_time = now
                    pkt = pack_set_position_target(
                        seq=self.seq,
                        vx=0.0,
                        vy=0.0,
                        vz=0.0,
                        yaw_rate=0.0,
                    )
                    self.seq = (self.seq + 1) & 0xFF
                    self.send_mavlink_packet(pkt)


# ============================================================
# Live MJPEG Streamer & Target Locking API (Ground Station Broadcast)
# ============================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BeaconBroadcaster:
    """
    Periodically broadcasts a UDP discovery beacon on port 14555.
    The Flutter app listens for this packet to instantly auto-detect the Pi's exact IP address.
    """
    def __init__(self, beacon_port=14555, stream_port=8080):
        self.beacon_port = beacon_port
        self.stream_port = stream_port
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="ASV-Beacon", daemon=True)
        self.thread.start()

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = f"ASV_PI5_STREAM:{self.stream_port}".encode("utf-8")
        while self.running:
            try:
                # Broadcast on global and Pi Hotspot (10.42.0.255) subnets
                sock.sendto(msg, ("255.255.255.255", self.beacon_port))
                sock.sendto(msg, ("10.42.0.255", self.beacon_port))
            except Exception:
                pass
            time.sleep(2.0)
        sock.close()

    def stop(self):
        self.running = False


class MJPEGStreamer:
    """
    Lightweight, multi-client HTTP MJPEG video streamer and target lock-on server for Ground Station.
    Streams annotated frames (camera feed + HUD + bounding boxes + reticle)
    over Pi 5 Wi-Fi Hotspot to Flutter ground station at low latency, and handles Click-to-Lock requests.
    """
    def __init__(self, host="0.0.0.0", port=8080, follow_controller=None):
        self.host = host
        self.port = port
        self.follow_controller = follow_controller
        self.latest_frame = None
        self.condition = threading.Condition()
        self.server = None
        self.thread = None
        self.beacon = None
        self.running = False

    def set_follow_controller(self, controller):
        self.follow_controller = controller

    def update_frame(self, frame_bytes: bytes):
        with self.condition:
            self.latest_frame = frame_bytes
            self.condition.notify_all()

    def start(self):
        streamer_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                # Print lock/unlock requests to terminal so operator sees phone interaction
                if len(args) > 0 and any(k in str(args[0]) for k in ("/lock", "/unlock")):
                    print(f"[HTTP] 🌐 {args[0]}")

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path in ("/", "/video_feed", "/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    try:
                        while streamer_ref.running:
                            with streamer_ref.condition:
                                streamer_ref.condition.wait(timeout=0.1)
                                frame = streamer_ref.latest_frame
                            if frame is None:
                                continue
                            header = (
                                b"--FRAME\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                            )
                            self.wfile.write(header)
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif path in ("/lock", "/api/lock", "/target_lock"):
                    params = parse_qs(parsed.query)
                    nx = params.get("nx", [None])[0]
                    ny = params.get("ny", [None])[0]
                    x = params.get("x", [None])[0]
                    y = params.get("y", [None])[0]
                    tid = params.get("track_id", [None])[0]
                    act = params.get("action", [None])[0]

                    res = handle_lock_request(nx=nx, ny=ny, x=x, y=y, track_id=tid, action=act)
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    self.close_connection = True
                elif path in ("/unlock", "/api/unlock"):
                    res = handle_lock_request(action="unlock")
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    self.close_connection = True
                elif path in ("/status", "/lock_status", "/api/status"):
                    with target_lock_mutex:
                        fc = streamer_ref.follow_controller or ACTIVE_FOLLOW_CONTROLLER
                        res = {
                            "locked": selected_id is not None,
                            "selected_id": selected_id,
                            "target_count": len(current_boxes),
                            "following": fc.is_following if fc else False,
                            "follow_status": fc.status if fc else ("LOCKED" if selected_id is not None else "STANDBY"),
                            "vx": round(fc.current_vx, 2) if fc else 0.0,
                            "yaw_rate": round(fc.current_yaw_rate, 2) if fc else 0.0,
                            "targets": [
                                {"track_id": item[1], "box": list(item[0])}
                                for item in current_boxes
                            ],
                        }
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.send_error(404)
                    self.end_headers()
                    self.close_connection = True

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path in ("/lock", "/api/lock", "/target_lock", "/unlock"):
                    content_len = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                    try:
                        data_in = json.loads(body.decode("utf-8")) if body else {}
                    except Exception:
                        data_in = {}

                    act = data_in.get("action")
                    if path.endswith("/unlock"):
                        act = "unlock"

                    res = handle_lock_request(
                        nx=data_in.get("nx"),
                        ny=data_in.get("ny"),
                        x=data_in.get("x"),
                        y=data_in.get("y"),
                        track_id=data_in.get("track_id"),
                        action=act,
                    )
                    data = json.dumps(res).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.send_error(404)
                    self.end_headers()
                    self.close_connection = True

        try:
            self.server = ThreadedHTTPServer((self.host, self.port), Handler)
            self.running = True
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

            # Start Auto-Discovery UDP Beacon
            self.beacon = BeaconBroadcaster(beacon_port=14555, stream_port=self.port)
            self.beacon.start()

            print("=" * 60)
            print(f"[STREAM] 📡 Live MJPEG Video Stream active at: http://0.0.0.0:{self.port}/video_feed")
            print(f"[STREAM] 📡 Auto-Discovery Beacon broadcasting on UDP 14555")
            print("=" * 60)
        except Exception as exc:
            print(f"[WARN] Failed to start MJPEG streamer on port {self.port}: {exc}")

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.beacon:
            try:
                self.beacon.stop()
            except Exception:
                pass
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


# ============================================================
# Unified Hailo / PyTorch Model Wrapper
# ============================================================

class ModelEngine:
    """
    Unified detector engine for Raspberry Pi 5.
    1. Prioritizes Hailo-8L NPU acceleration via Ultralytics (best.hef).
    2. Falls back to native hailo_platform if Ultralytics HEF loader has version issues.
    3. Falls back to PyTorch (best.pt) on CPU if Hailo hardware is unavailable.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.is_hef = model_path.lower().endswith(".hef")
        self.native_hailo = False
        self.backend_name = "HAILO-8L (NativeRT)" if self.is_hef else "Ultralytics"
        self.names = {0: "person", 1: "vehicle"}

        # Use native HailoRT directly for HEF files.
        if self.is_hef:
            print("=" * 60)
            print(f"[ENGINE] Loading Hailo HEF: {model_path}")
            print("[INFO] Initializing native HailoRT...")
            self._init_native_hailo(model_path)
            return

        # Use Ultralytics for supported model formats, such as .pt.
        print("=" * 60)
        print(f"[ENGINE] Loading Ultralytics model: {model_path}")

        self.model = YOLO(model_path)

        if self.model.names:
            self.names = self.model.names

        self.backend_name = "Ultralytics"
        print("[SUCCESS] Ultralytics model initialized.")
        print("=" * 60)

    def _init_native_hailo(self, hef_path):
        """Initialize native HailoRT for a PCIe-connected Hailo-8L."""
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
                InferVStreams,
            )

            # Load HEF model and initialize the Hailo device.
            self.hef = HEF(hef_path)
            self.vdevice = VDevice()

            # Configure the Hailo device using the PCIe interface.
            configure_params = ConfigureParams.create_from_hef(
                self.hef,
                interface=HailoStreamInterface.PCIe,
            )

            network_groups = self.vdevice.configure(
                self.hef, configure_params
            )
            self.network_group = network_groups[0]
            # Activate the Hailo network group before inference.
            self.network_group_params = self.network_group.create_params()
            self.activation_context = self.network_group.activate(
                self.network_group_params
            )
            self.activation_context.__enter__()

            # Get model input and output information.
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_infos = self.hef.get_output_vstream_infos()

            print(f"[INFO] Input name: {self.input_vstream_info.name}")
            print(f"[INFO] Input shape: {self.input_vstream_info.shape}")

            for output_info in self.output_vstream_infos:
                print(
                    f"[INFO] Output name: {output_info.name}, "
                    f"shape: {output_info.shape}"
                )

            # Configure input and output streams.
            self.input_params = InputVStreamParams.make(
                self.network_group,
                format_type=FormatType.UINT8,
            )

            self.output_params = OutputVStreamParams.make(
                self.network_group,
                format_type=FormatType.FLOAT32,
            )

            # Create the inference pipeline.
            self.infer_pipeline = InferVStreams(
                self.network_group,
                self.input_params,
                self.output_params,
            )

            self.infer_pipeline.__enter__()

            self.native_hailo = True
            self.backend_name = "HAILO-8L (NativeRT)"

            print("=" * 60)
            print("[SUCCESS] Native HailoRT initialized successfully.")
            print("[INFO] Hailo-8L inference backend is ready.")
            print("=" * 60)

        except Exception as exc:
            print(f"[ERROR] Failed to initialize native HailoRT: {exc}")

            # Release the device if initialization failed partway through.
            if hasattr(self, "vdevice"):
                try:
                    self.vdevice.release()
                except Exception:
                    pass

            raise

    def predict(self, frame, conf=0.05, imgsz=640, use_letterbox=True):
        """Returns list of detections: [x1, y1, x2, y2, conf, cls]"""
        detections = []

        if not self.native_hailo:
            # Primary Ultralytics prediction pipeline
            results = self.model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = to_numpy(result.boxes.xyxy)
                confs = to_numpy(result.boxes.conf)
                classes = to_numpy(result.boxes.cls)

                for box, score, cls in zip(boxes, confs, classes):
                    detections.append([
                        float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                        float(score), int(cls)
                    ])
            return detections

        # Fallback Native HailoRT prediction pipeline
        orig_h, orig_w = frame.shape[:2]
        if use_letterbox:
            resized, r, (dw, dh) = letterbox(frame, new_shape=(imgsz, imgsz))
        else:
            resized = cv2.resize(frame, (imgsz, imgsz))
            r = imgsz / max(orig_w, 1)
            dw, dh = 0.0, 0.0

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = {self.input_vstream_info.name: np.expand_dims(rgb, axis=0)}

        raw_results = self.infer_pipeline.infer(input_data)

        for _, output in raw_results.items():
            if isinstance(output, (list, tuple)):
                if len(output) > 0 and isinstance(output[0], (list, tuple)):
                    class_outputs = output[0]
                else:
                    class_outputs = output
            else:
                class_outputs = [output]

            for real_cls_idx, class_output in enumerate(class_outputs):
                if isinstance(class_output, (list, tuple)):
                    for nested_output in class_output:
                        try:
                            boxes = np.asarray(nested_output)
                        except (ValueError, TypeError):
                            continue

                        if boxes.size == 0:
                            continue

                        if boxes.ndim == 1:
                            boxes = boxes.reshape(1, -1)

                        if boxes.ndim != 2 or boxes.shape[1] < 5:
                            continue

                        for detection in boxes:
                            y1, x1, y2, x2, score = detection[:5]

                            if score < conf:
                                continue

                            # Hailo NMS coordinates are normally pixel
                            # coordinates, not normalized coordinates.
                            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                                x1 *= imgsz
                                x2 *= imgsz
                                y1 *= imgsz
                                y2 *= imgsz

                            if use_letterbox:
                                x1 = (x1 - dw) / r
                                x2 = (x2 - dw) / r
                                y1 = (y1 - dh) / r
                                y2 = (y2 - dh) / r
                            else:
                                x1 *= orig_w / imgsz
                                x2 *= orig_w / imgsz
                                y1 *= orig_h / imgsz
                                y2 *= orig_h / imgsz

                            x1 = max(0.0, min(float(orig_w), float(x1)))
                            x2 = max(0.0, min(float(orig_w), float(x2)))
                            y1 = max(0.0, min(float(orig_h), float(y1)))
                            y2 = max(0.0, min(float(orig_h), float(y2)))

                            detections.append([
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2),
                                float(score),
                                int(real_cls_idx),
                            ])
                else:
                    try:
                        boxes = np.asarray(class_output)
                    except (ValueError, TypeError):
                        continue

                    if boxes.size == 0:
                        continue

                    if boxes.ndim == 1:
                        boxes = boxes.reshape(1, -1)

                    if boxes.ndim != 2 or boxes.shape[1] < 5:
                        continue

                    for detection in boxes:
                        y1, x1, y2, x2, score = detection[:5]

                        if score < conf:
                            continue

                        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                            x1 *= imgsz
                            x2 *= imgsz
                            y1 *= imgsz
                            y2 *= imgsz

                        if use_letterbox:
                            x1 = (x1 - dw) / r
                            x2 = (x2 - dw) / r
                            y1 = (y1 - dh) / r
                            y2 = (y2 - dh) / r
                        else:
                            x1 *= orig_w / imgsz
                            x2 *= orig_w / imgsz
                            y1 *= orig_h / imgsz
                            y2 *= orig_h / imgsz

                        x1 = max(0.0, min(float(orig_w), float(x1)))
                        x2 = max(0.0, min(float(orig_w), float(x2)))
                        y1 = max(0.0, min(float(orig_h), float(y1)))
                        y2 = max(0.0, min(float(orig_h), float(y2)))

                        detections.append([
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            float(score),
                            int(real_cls_idx),
                        ])

        return detections


    def close(self):
        """Cleanup Hailo device resources."""
        if self.native_hailo:
            try:
                if hasattr(self, "infer_pipeline"):
                    self.infer_pipeline.__exit__(None, None, None)

                if hasattr(self, "activation_context"):
                    self.activation_context.__exit__(None, None, None)

                if hasattr(self, "vdevice"):
                    self.vdevice.release()

            except Exception as exc:
                print(f"[WARN] Hailo cleanup: {exc}")


# ============================================================
# Mouse & HUD Telemetry
# ============================================================

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        handle_lock_request(x=x, y=y)


def draw_hud(frame, selected_id, track_count, fps=0.0, backend_name="HAILO-8L", follow_controller=None):
    """Draw tactical drone HUD telemetry header and visual servoing tracking vectors."""
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # 1. Subtle Center Optical Crosshairs
    crosshair_color = (180, 180, 180)
    cv2.circle(frame, (cx, cy), 5, crosshair_color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 14, cy), (cx - 6, cy), crosshair_color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 6, cy), (cx + 14, cy), crosshair_color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 14), (cx, cy - 6), crosshair_color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 6), (cx, cy + 14), crosshair_color, 1, cv2.LINE_AA)

    # 2. Visual Servoing Target Lead Vector
    if follow_controller is not None and selected_id is not None and follow_controller.target_center is not None:
        tx, ty = follow_controller.target_center
        vector_color = (0, 0, 255) if follow_controller.is_following else (0, 215, 255)
        cv2.line(frame, (cx, cy), (tx, ty), vector_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), 4, vector_color, -1, cv2.LINE_AA)

        # Offset text near midpoint
        mx, my = (cx + tx) // 2, (cy + ty) // 2
        offset_text = f"dX: {tx - cx:+d}px"
        cv2.putText(
            frame, offset_text, (mx + 8, my - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, vector_color, 1, cv2.LINE_AA
        )

    # 3. Telemetry bar overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Telemetry info (Left)
    telemetry_left = f"ASV-X1 [{backend_name}] | {fps:.1f} FPS | TARGETS: {track_count}"
    cv2.putText(
        frame, telemetry_left, (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Telemetry info (Right)
    if selected_id is not None:
        if follow_controller is not None and follow_controller.is_following:
            vx = follow_controller.current_vx
            yaw_deg = math.degrees(follow_controller.current_yaw_rate)
            target_text = f"LOCKED #{selected_id} • FOLLOWING [Vx: {vx:.1f}m/s, Yaw: {yaw_deg:+.0f}°/s]"
            color = (0, 255, 120)
        elif follow_controller is not None and "HOLDING" in follow_controller.status:
            target_text = f"LOCKED #{selected_id} • HOLDING HOVER"
            color = (0, 215, 255)
        else:
            target_text = f"LOCK-ON: ID #{selected_id}"
            color = (0, 0, 255)
    else:
        target_text = "LOCK-ON: NONE (CLICK TO LOCK)"
        color = (0, 255, 255)

    (txt_w, _), _ = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(
        frame, target_text, (w - txt_w - 20, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1 if selected_id is None else 2, cv2.LINE_AA
    )



def draw_target_reticle(frame, x1, y1, x2, y2, color=(0, 0, 255)):
    """Draw tactical lock-on reticle corners around selected target."""
    length = max(14, int(min(x2 - x1, y2 - y1) * 0.25))

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, 3)

    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, 3)

    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, 3)

    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, 3)


# ============================================================
# Camera Initialization
# ============================================================

def open_camera():
    """Open Raspberry Pi Camera via Picamera2 (libcamera)."""
    if not PICAMERA2_AVAILABLE:
        print("[ERROR] Picamera2 is not available.")
        print("[ERROR] Install it with: sudo apt install -y python3-picamera2")
        return None

    print("[INFO] Initializing Raspberry Pi Camera via Picamera2...")
    picam2 = Picamera2()
    # Note: In libcamera/DRM fourcc convention, format 'RGB888' is stored in
    # memory as [B, G, R], which is already the native BGR order expected by OpenCV.
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print("[INFO] Raspberry Pi Camera started: 1280x720 (Native OpenCV BGR order)")
    return picam2


# ============================================================
# Model Path Resolution
# ============================================================

def resolve_model_path(requested_path):
    """Automatically prioritizes Hailo HEF model, then falls back to PyTorch."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if requested_path and requested_path != "auto":
        if os.path.exists(requested_path):
            return requested_path
        alt = os.path.join(base_dir, requested_path)
        if os.path.exists(alt):
            return alt

    # Priority 1: Hailo HEF model (26 TOPS NPU)
    hef_candidates = [
        os.path.join(base_dir, "model", "best.hef"),
        os.path.join(base_dir, "best.hef"),
        "model/best.hef",
        "best.hef",
    ]
    for path in hef_candidates:
        if os.path.exists(path):
            print("=" * 60)
            print(f"[ACCELERATION] Found Hailo HEF model: {path}")
            print(f"[ACCELERATION] Using Hailo-8L 26 TOPS AI Hat Hardware Acceleration!")
            print("=" * 60)
            return path

    # Priority 2: PyTorch model (Fallback to CPU)
    pt_candidates = [
        os.path.join(base_dir, "model", "best.pt"),
        os.path.join(base_dir, "best.pt"),
        "model/best.pt",
        "best.pt",
    ]
    for path in pt_candidates:
        if os.path.exists(path):
            print("=" * 60)
            print(f"[WARN] Hailo .hef model not found. Falling back to PyTorch CPU model: {path}")
            print("=" * 60)
            return path

    return requested_path


# ============================================================
# Main Tracking Pipeline
# ============================================================

def track(
    video_path="0",
    model_path="auto",
    swap_rb=False,
    letterbox=True,
    stream=True,
    stream_port=8080,
    stream_quality=65,
    headless=False,
    router=None,
    enable_follow=True,
    max_speed=1.5,
    max_yaw_rate=0.70,
):
    global current_boxes, selected_id, current_frame_shape, ACTIVE_FOLLOW_CONTROLLER

    # Initialize Autonomous Follow Controller
    follow_controller = None
    if enable_follow:
        follow_controller = DroneFollowController(
            router=router,
            mav_udp_port=14550,
            max_speed=max_speed,
            max_yaw_rate=max_yaw_rate,
        )
        ACTIVE_FOLLOW_CONTROLLER = follow_controller
        print("=" * 60)
        print("[AUTONOMOUS FOLLOW] 🛰️ Visual Servoing Follow Controller Active")
        print(f"[AUTONOMOUS FOLLOW] Max Speed: {max_speed:.1f} m/s | Max Yaw Rate: {max_yaw_rate:.2f} rad/s (~{math.degrees(max_yaw_rate):.0f}°/s)")
        print("[AUTONOMOUS FOLLOW] Guided Mode & Body-NED velocity control ready.")
        print("=" * 60)

    # Start live MJPEG video streamer for Ground Station
    streamer = None
    if stream:
        try:
            streamer = MJPEGStreamer(host="0.0.0.0", port=stream_port, follow_controller=follow_controller)
            streamer.start()
        except Exception as exc:
            print(f"[WARN] Failed to start video streamer: {exc}")

    resolved_model = resolve_model_path(model_path)
    if not resolved_model or not os.path.exists(resolved_model):
        print(f"[ERROR] Model file not found: {model_path} (Resolved: {resolved_model})")
        return

    # Initialize Engine (Hailo-8L or PyTorch)
    engine = ModelEngine(resolved_model)

    # Determine Video Source
    try:
        video_src = int(video_path)
    except ValueError:
        video_src = video_path

    picam2 = None
    cap = None
    use_picamera = False

    if video_src == 0:
        # Priority: Picamera2 for Raspberry Pi CSI ribbon camera
        picam2 = open_camera()
        if picam2 is None:
            print("[WARN] Picamera2 unavailable. Falling back to standard OpenCV capture(0)...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Could not open camera device 0.")
                engine.close()
                return
        else:
            use_picamera = True
    else:
        # Video file or external USB camera node
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {video_src}")
            engine.close()
            return

    # Initialize Drone Tracker with Camera Motion Compensation
    tracker = DroneTracker(
        det_thresh=0.20,
        low_thresh=0.05,
        match_thresh=0.80,
        max_age=35,
        min_hits=2,
        use_gmc=True
    )

    window_name = "ASV-X1 Professional Drone Tracking (Raspberry Pi 5 + Hailo-8L)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        print("[INFO] Drone tracking started. Press 'q' to quit, click target to lock-on.")
    else:
        print("[INFO] Drone tracking started (headless mode).")

    fps = 0.0
    fps_history = []
    t_prev = time.time()

    try:
        while True:
            # 1. Capture Frame
            if use_picamera:
                # Picamera2 configured with 'RGB888' returns [B, G, R] byte layout in memory,
                # which is directly the native BGR format OpenCV expects.
                frame = picam2.capture_array()
                if frame is None:
                    print("[ERROR] Failed to capture frame from Picamera2.")
                    break
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                ret = True
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                print("[INFO] End of video feed.")
                break

            if swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. Resize for Display & Processing
            frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            # 3. Hardware-Accelerated Detection via Hailo-8L
            detections = engine.predict(frame, conf=0.05, imgsz=640, use_letterbox=letterbox)

            # 4. Multi-Object Tracking & Re-ID Update
            active_tracks = tracker.update(detections, frame)

            new_boxes = []
            for track_obj in active_tracks:
                track_id = track_obj.track_id
                x1, y1, x2, y2 = map(int, track_obj.to_tlbr())
                new_boxes.append(((x1, y1, x2, y2), track_id))

            h, w = frame.shape[:2]
            with target_lock_mutex:
                current_frame_shape = (h, w)
                current_boxes = new_boxes

            # 5. Autonomous Follow Controller Update (Visual Servoing)
            if follow_controller is not None:
                follow_controller.update(selected_id, active_tracks, frame.shape)

            # 6. Draw Tracks
            for track_obj in active_tracks:
                track_id = track_obj.track_id
                x1, y1, x2, y2 = map(int, track_obj.to_tlbr())
                cls_id = track_obj.cls
                conf = track_obj.conf

                is_selected = (track_id == selected_id)
                color = (0, 0, 255) if is_selected else (0, 255, 0)
                thickness = 3 if is_selected else 1

                # Trajectory Trail
                if len(track_obj.history) > 1:
                    pts = np.array(track_obj.history, dtype=np.int32).reshape((-1, 1, 2))
                    trail_color = (50, 50, 255) if is_selected else (0, 200, 100)
                    cv2.polylines(frame, [pts], False, trail_color, thickness=2 if is_selected else 1)

                # Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Lock-on Reticle
                if is_selected:
                    draw_target_reticle(frame, x1, y1, x2, y2, color=(0, 0, 255))

                # Label Header
                class_name = engine.names.get(cls_id, f"C:{cls_id}")
                label = f"ID:{track_id} | {class_name} {conf:.2f}"

                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                label_y1 = max(0, y1 - txt_h - 6)
                label_y2 = max(txt_h + 6, y1)

                cv2.rectangle(frame, (x1, label_y1), (x1 + txt_w + 4, label_y2), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, label_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

            # 7. FPS Calculation
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 20:
                    fps_history.pop(0)
                fps = sum(fps_history) / len(fps_history)

            # 8. Draw Target & Tracker HUD (Top)
            draw_hud(frame, selected_id, len(active_tracks), fps=fps, backend_name=engine.backend_name, follow_controller=follow_controller)

            # 9. Broadcast live annotated frame to Ground Station (MJPEG)
            if streamer is not None:
                success, enc_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, stream_quality])
                if success:
                    streamer.update_frame(enc_frame.tobytes())

            if not headless:
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        # Cleanup Resources
        if follow_controller is not None:
            follow_controller.send_stop(count=5)
        ACTIVE_FOLLOW_CONTROLLER = None
        if streamer is not None:
            streamer.stop()
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        engine.close()
        if not headless:
            cv2.destroyAllWindows()
        print("[INFO] Tracking shut down cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASV-X1 Professional Drone Tracking (Raspberry Pi 5 + Hailo-8L AI Hat)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 = Raspberry Pi Camera (Picamera2) / USB Camera, or path to video file."
    )
    parser.add_argument(
        "--model", type=str, default="auto",
        help="Path to YOLO model (default: 'auto', detects model/best.hef for Hailo-8L 26 TOPS NPU)."
    )
    parser.add_argument(
        "--swap-rb", action="store_true",
        help="Manually invert Red and Blue channels if your camera sensor produces swapped colors."
    )
    parser.add_argument(
        "--no-letterbox", action="store_true",
        help="Disable aspect-ratio letterboxing (defaults to letterbox enabled)."
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable live MJPEG video streaming to Ground Station."
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080,
        help="Port for live MJPEG video stream (default: 8080)."
    )
    parser.add_argument(
        "--stream-quality", type=int, default=65,
        help="JPEG compression quality for streaming (1-100, default: 65, optimized for Pi 5 Wi-Fi)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV GUI window."
    )
    parser.add_argument(
        "--no-follow", action="store_true",
        help="Disable autonomous drone target following when locked."
    )
    parser.add_argument(
        "--max-speed", type=float, default=1.5,
        help="Maximum forward follow velocity in m/s (default: 1.5 m/s)."
    )
    parser.add_argument(
        "--max-yaw-rate", type=float, default=0.70,
        help="Maximum yaw turning rate in rad/s (default: 0.70 rad/s = ~40 deg/s)."
    )

    args = parser.parse_args()
    track(
        args.source,
        args.model,
        swap_rb=args.swap_rb,
        letterbox=not args.no_letterbox,
        stream=not args.no_stream,
        stream_port=args.stream_port,
        stream_quality=args.stream_quality,
        headless=args.headless,
        enable_follow=not args.no_follow,
        max_speed=args.max_speed,
        max_yaw_rate=args.max_yaw_rate,
    )

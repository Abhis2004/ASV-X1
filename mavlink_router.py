"""
ASV-X1 USB MAVLink Router (Raspberry Pi 5 ↔ Pixhawk ↔ Wi-Fi Ground Station)
Routes Pixhawk MAVLink packets directly over a standard USB serial connection (/dev/ttyUSB0 or /dev/ttyACM0)
and broadcasts them bi-directionally over both UDP (port 14550) and TCP (port 5760).

Key Features:
- Dual-Protocol: UDP 14550 (Standard Broadcast & Unicast) + TCP 5760 (Reliable Stream for Ground Station App)
- Auto-Detects Pixhawk USB serial port (/dev/ttyUSB0, /dev/ttyACM*, /dev/serial/by-id/*)
- Thread-safe serial write lock (prevents packet slicing across keepalive, UDP, and TCP threads)
- Zero-lock non-blocking serial reads
- Instant Stream Activation: immediately requests telemetry streams when a ground station connects (TCP or UDP)
- Continuous GCS Heartbeat keepalives (1 Hz) to prevent ArduPilot telemetry timeout
- Auto-Discovers connected hotspot clients from /proc/net/arp to bypass Android UDP broadcast filters
- Live 3-second throughput reporting in journalctl (Rx from Pixhawk / Tx to Clients)
"""

import os
import sys
import glob
import time
import socket
import select
import threading
from typing import Optional, List, Set, Tuple

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


def find_pixhawk_port() -> Optional[str]:
    """Auto-detect Pixhawk USB serial port across common Linux device nodes and persistent links."""
    # 1. Standard Linux USB-to-UART converters (CP2102, FTDI, CH340 used for TELEM2)
    for p in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"]:
        if os.path.exists(p):
            return p

    # 2. Standard Linux CDC-ACM USB serial devices (Pixhawk native USB)
    for p in ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]:
        if os.path.exists(p):
            return p

    # 3. Search persistent serial by-id symlinks
    by_id_ports = glob.glob("/dev/serial/by-id/*")
    for p in by_id_ports:
        p_lower = p.lower()
        if any(keyword in p_lower for keyword in (
            "ardupilot", "px4", "pixhawk", "stm32", "cube",
            "cp210", "ftdi", "ch340", "usb_serial", "usb-serial", "uart"
        )):
            return p

    if len(by_id_ports) == 1:
        return by_id_ports[0]

    return None


def get_connected_hotspot_ips() -> List[str]:
    """Auto-detects IP addresses of client devices (phones) connected to the Pi hotspot."""
    ips = set()
    try:
        if os.path.exists("/proc/net/arp"):
            with open("/proc/net/arp", "r") as f:
                for line in f.readlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 6:
                        ip = parts[0]
                        flags = parts[2]
                        if flags != "0x0" and ip.startswith("10.42.0."):
                            ips.add(ip)
    except Exception:
        pass

    for lf in ["/var/lib/misc/dnsmasq.leases", "/var/lib/NetworkManager/dnsmasq-wlan0.leases"]:
        try:
            if os.path.exists(lf):
                with open(lf, "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 3 and parts[2].startswith("10.42.0."):
                            ips.add(parts[2])
        except Exception:
            pass

    return list(ips)


ACTIVE_ROUTER: Optional["MavlinkRouter"] = None


def get_active_router() -> Optional["MavlinkRouter"]:
    return ACTIVE_ROUTER


class MavlinkRouter:
    def __init__(
        self,
        port: str = "auto",
        baudrate: int = 115200,
        udp_port: int = 14550,
        tcp_port: int = 5760,
        broadcast_ips: Optional[List[str]] = None,
        buffer_size: int = 4096,
    ):
        self.requested_port = port
        self.baudrate = baudrate
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.buffer_size = buffer_size

        self.broadcast_ips = broadcast_ips or [
            "255.255.255.255",
            "10.42.0.255",
        ]

        self.known_udp_clients: Set[Tuple[str, int]] = set()
        self.active_tcp_clients: Set[socket.socket] = set()
        self.clients_lock = threading.Lock()
        self.ser_write_lock = threading.Lock()

        self.running = False
        self.ser: Optional[serial.Serial] = None
        self.udp_sock: Optional[socket.socket] = None
        self.tcp_server_sock: Optional[socket.socket] = None

        self.active_port: Optional[str] = None
        self.bytes_from_serial = 0
        self.bytes_from_udp = 0
        self.bytes_from_tcp = 0

        self._last_open_err_time = 0.0

        # Baud rate candidates: prioritize requested baudrate (115200), with 57600 fallback
        if self.baudrate == 115200:
            self.baud_candidates = [115200, 57600]
        elif self.baudrate == 57600:
            self.baud_candidates = [57600, 115200]
        else:
            self.baud_candidates = [self.baudrate, 115200, 57600]
        self.baud_idx = 0
        self.current_baud = self.baud_candidates[0]

        # Health watchdog & MAVLink stream validation state
        self.last_valid_mavlink_time = 0.0
        self.ser_open_time = 0.0
        self.sync_locked = False
        self.valid_mavlink_packets = 0

        self.vehicle_type = 2
        self.custom_mode = 0
        self.is_armed = False

    def send_to_serial(self, data: bytes) -> bool:
        """Thread-safe serial write."""
        if self.ser and self.ser.is_open:
            try:
                with self.ser_write_lock:
                    self.ser.write(data)
                return True
            except Exception:
                pass
        return False

    def trigger_stream_request(self):
        """Immediately pumps GCS heartbeat and stream requests to wake up Pixhawk telemetry."""
        hb_packet = bytes([254, 9, 0, 255, 190, 0, 0, 0, 0, 0, 6, 8, 0, 4, 3, 73, 33])
        stream_packets = [
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 0, 1, 151, 114]),   # ALL @ 4 Hz
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 1, 1, 75, 40]),    # RAW_SENSORS @ 4 Hz
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 2, 1, 47, 199]),   # EXTENDED_STATUS @ 4 Hz
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 6, 1, 78, 164]),   # POSITION @ 4 Hz
            bytes([254, 6, 0, 255, 190, 66, 10, 0, 1, 1, 10, 1, 204, 135]), # EXTRA1 (ATTITUDE) @ 10 Hz
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 11, 1, 49, 91]),   # EXTRA2 (VFR_HUD) @ 4 Hz
            bytes([254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 12, 1, 52, 215]),  # EXTRA3 @ 4 Hz
        ]
        self.send_to_serial(hb_packet)
        for sp in stream_packets:
            self.send_to_serial(sp)

    def start(self) -> bool:
        global ACTIVE_ROUTER
        if not SERIAL_AVAILABLE:
            print("[ERROR] 'pyserial' is not installed.")
            return False

        ACTIVE_ROUTER = self
        self.running = True

        try:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.udp_sock.bind(("0.0.0.0", self.udp_port))
            self.udp_sock.settimeout(0.3)
            print(f"[MAVLINK ROUTER] 📡 Bound UDP listener to 0.0.0.0:{self.udp_port}")
        except Exception as exc:
            print(f"[ERROR] Failed to bind UDP port {self.udp_port}: {exc}")
            return False

        try:
            self.tcp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_server_sock.bind(("0.0.0.0", self.tcp_port))
            self.tcp_server_sock.listen(5)
            self.tcp_server_sock.settimeout(1.0)
            print(f"[MAVLINK ROUTER] 🔌 Bound TCP server to 0.0.0.0:{self.tcp_port}")
        except Exception as exc:
            print(f"[WARN] TCP server bind notice (port {self.tcp_port}): {exc}")

        threading.Thread(target=self._serial_to_network_loop, name="Mavlink-Ser2Net", daemon=True).start()
        threading.Thread(target=self._udp_to_serial_loop, name="Mavlink-UDP2Ser", daemon=True).start()
        threading.Thread(target=self._stream_keepalive_loop, name="Mavlink-Keepalive", daemon=True).start()
        if self.tcp_server_sock:
            threading.Thread(target=self._tcp_accept_loop, name="Mavlink-TCPAccept", daemon=True).start()
        threading.Thread(target=self._hotspot_arp_scanner_loop, name="Mavlink-ARPScanner", daemon=True).start()
        threading.Thread(target=self._stats_reporter_loop, name="Mavlink-Stats", daemon=True).start()

        print("[MAVLINK ROUTER] 🚀 Multi-client router threads running.")
        return True

    def stop(self):
        global ACTIVE_ROUTER
        if ACTIVE_ROUTER is self:
            ACTIVE_ROUTER = None
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        if self.udp_sock:
            try:
                self.udp_sock.close()
            except Exception:
                pass
        if self.tcp_server_sock:
            try:
                self.tcp_server_sock.close()
            except Exception:
                pass
        with self.clients_lock:
            for s in list(self.active_tcp_clients):
                try:
                    s.close()
                except Exception:
                    pass
            self.active_tcp_clients.clear()
        print("[MAVLINK ROUTER] Stopped.")

    def _stats_reporter_loop(self):
        """Prints live router throughput stats to systemd journal every 3 seconds."""
        last_rx = 0
        while self.running:
            time.sleep(3.0)
            if self.ser and self.ser.is_open:
                rx_delta = self.bytes_from_serial - last_rx
                last_rx = self.bytes_from_serial
                total_kb = self.bytes_from_serial // 1024
                udp_c = len(self.known_udp_clients)
                with self.clients_lock:
                    tcp_c = len(self.active_tcp_clients)
                sync_tag = f"LOCKED ({self.valid_mavlink_packets} pkts)" if self.sync_locked else "SYNCING..."
                print(f"[MAVLINK ROUTER] 📊 Rx from Pixhawk: {total_kb} KB (+{rx_delta} B/3s) | {self.current_baud} baud [{sync_tag}] | UDP: {udp_c} | TCP: {tcp_c}", flush=True)

    def _open_serial(self) -> Optional[serial.Serial]:
        target_port = self.requested_port
        if not target_port or target_port == "auto" or not os.path.exists(target_port):
            detected = find_pixhawk_port()
            if detected:
                target_port = detected

        if not target_port or not os.path.exists(target_port):
            now = time.time()
            if now - self._last_open_err_time > 5.0:
                self._last_open_err_time = now
                print(f"[MAVLINK ROUTER] ⏳ Waiting for Pixhawk serial port (target: {self.requested_port or 'auto'})...")
            return None

        self.current_baud = self.baud_candidates[self.baud_idx % len(self.baud_candidates)]

        try:
            ser = serial.Serial(
                port=target_port,
                baudrate=self.current_baud,
                timeout=0.2,
                write_timeout=0.5,
                rtscts=False,
                dsrdtr=False,
            )
            # Only assert DTR/RTS for native Pixhawk USB CDC-ACM (ttyACM).
            # On USB-to-UART converters (ttyUSB), asserting DTR/RTS blocks UART framing!
            if "ttyACM" in target_port:
                ser.dtr = True
                ser.rts = True
            else:
                ser.dtr = False
                ser.rts = False

            ser.reset_input_buffer()
            ser.reset_output_buffer()

            self.active_port = target_port
            self.ser_open_time = time.time()
            self.last_valid_mavlink_time = time.time()
            self.sync_locked = False
            self.valid_mavlink_packets = 0

            print(f"[MAVLINK ROUTER] 🔌 Connected to Pixhawk USB: {target_port} @ {self.current_baud} baud")

            self.ser = ser
            self.trigger_stream_request()
            return ser
        except Exception as exc:
            now = time.time()
            if now - self._last_open_err_time > 5.0:
                self._last_open_err_time = now
                print(f"[MAVLINK ROUTER] ⚠️ Could not open {target_port}: {exc}. Retrying...")
            self.active_port = None
            return None

    def _stream_keepalive_loop(self):
        hb_packet = bytes([254, 9, 0, 255, 190, 0, 0, 0, 0, 0, 6, 8, 0, 4, 3, 73, 33])
        count = 0
        while self.running:
            time.sleep(1.0)
            if self.ser and self.ser.is_open:
                self.send_to_serial(hb_packet)
                count += 1
                if count >= 3:
                    count = 0
                    self.trigger_stream_request()

    def _hotspot_arp_scanner_loop(self):
        while self.running:
            time.sleep(2.5)
            client_ips = get_connected_hotspot_ips()
            for ip in client_ips:
                endpoint = (ip, self.udp_port)
                if endpoint not in self.known_udp_clients:
                    with self.clients_lock:
                        self.known_udp_clients.add(endpoint)
                    print(f"[MAVLINK ROUTER] 📱 Discovered hotspot phone at {ip} (Added to direct UDP telemetry)")
                    self.trigger_stream_request()

    def _serial_to_network_loop(self):
        while self.running:
            if self.ser is None or not self.ser.is_open:
                self.ser = self._open_serial()
                if self.ser is None:
                    time.sleep(1.0)
                    continue

            now = time.time()
            time_open = now - self.ser_open_time
            time_since_valid = now - self.last_valid_mavlink_time

            # Serial stream watchdog: if no valid MAVLink packets arrive, auto-cycle and adjust baud rate
            if (time_open > 2.5 and not self.sync_locked) or (self.sync_locked and time_since_valid > 4.5):
                print(
                    f"[MAVLINK ROUTER] ⚠️ Serial link desync at {self.current_baud} baud "
                    f"(no valid MAVLink headers for {time_since_valid:.1f}s). Re-syncing port...",
                    flush=True
                )
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                self.sync_locked = False
                self.baud_idx += 1  # Cycle candidate baud rate (115200 <-> 57600)
                time.sleep(0.5)
                continue

            try:
                in_wait = self.ser.in_waiting
                if in_wait > 0:
                    data = self.ser.read(min(in_wait, self.buffer_size))
                else:
                    data = self.ser.read(1)

                if not data:
                    time.sleep(0.01)
                    continue

                # Verify MAVLink 1 (0xFE / 254) or MAVLink 2 (0xFD / 253) framing
                has_magic = (b'\xfe' in data) or (b'\xfd' in data)
                if has_magic:
                    self.last_valid_mavlink_time = time.time()
                    self.valid_mavlink_packets += data.count(b'\xfe') + data.count(b'\xfd')
                    if not self.sync_locked:
                        self.sync_locked = True
                        print(
                            f"[MAVLINK ROUTER] ✅ MAVLink sync verified at {self.current_baud} baud! "
                            f"(Lock established on {self.active_port})",
                            flush=True
                        )

                # Filter out pure framing noise / doubled-bit artifacts while desynced
                if not self.sync_locked and not has_magic:
                    continue

                self.bytes_from_serial += len(data)

                with self.clients_lock:
                    targets = list(self.known_udp_clients)

                for client_addr in targets:
                    try:
                        self.udp_sock.sendto(data, client_addr)
                    except Exception:
                        pass

                for bcast_ip in self.broadcast_ips:
                    try:
                        self.udp_sock.sendto(data, (bcast_ip, self.udp_port))
                    except Exception:
                        pass

                dead_tcp = []
                with self.clients_lock:
                    tcp_sockets = list(self.active_tcp_clients)

                for s in tcp_sockets:
                    try:
                        s.sendall(data)
                    except Exception:
                        dead_tcp.append(s)

                if dead_tcp:
                    with self.clients_lock:
                        for s in dead_tcp:
                            self.active_tcp_clients.discard(s)
                            try:
                                s.close()
                            except Exception:
                                pass

            except (serial.SerialException, OSError) as exc:
                print(f"[WARN] Pixhawk USB disconnected: {exc}. Reconnecting...")
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
                self.sync_locked = False
                time.sleep(0.5)

    def _udp_to_serial_loop(self):
        while self.running:
            if self.udp_sock is None:
                time.sleep(0.5)
                continue

            try:
                data, addr = self.udp_sock.recvfrom(self.buffer_size)
                if not data:
                    continue

                if addr[0] in ("127.0.0.1", "10.42.0.1"):
                    continue

                with self.clients_lock:
                    if addr not in self.known_udp_clients:
                        self.known_udp_clients.add(addr)
                        print(f"[MAVLINK ROUTER] 📱 UDP Ground Station linked: {addr[0]}:{addr[1]}")
                        self.trigger_stream_request()

                self.bytes_from_udp += len(data)
                self.send_to_serial(data)

            except socket.timeout:
                continue
            except Exception:
                if self.running:
                    time.sleep(0.1)

    def _tcp_accept_loop(self):
        while self.running and self.tcp_server_sock:
            try:
                client_sock, client_addr = self.tcp_server_sock.accept()
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_sock.settimeout(None)
                client_sock.setblocking(True)
                with self.clients_lock:
                    self.active_tcp_clients.add(client_sock)
                print(f"[MAVLINK ROUTER] ⚡ TCP Ground Station connected: {client_addr[0]}:{client_addr[1]}")
                self.trigger_stream_request()

                threading.Thread(
                    target=self._tcp_client_read_loop,
                    args=(client_sock,),
                    name=f"TCP-Client-{client_addr[0]}",
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def _tcp_client_read_loop(self, client_sock: socket.socket):
        while self.running:
            try:
                data = client_sock.recv(self.buffer_size)
                if not data:
                    break
                self.bytes_from_tcp += len(data)
                self.send_to_serial(data)
            except Exception:
                break

        with self.clients_lock:
            self.active_tcp_clients.discard(client_sock)
        try:
            client_sock.close()
        except Exception:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ASV-X1 USB MAVLink Router (Pixhawk USB ↔ Pi 5 Hotspot UDP 14550 & TCP 5760)"
    )
    parser.add_argument("--port", type=str, default="auto", help="Pixhawk serial port.")
    parser.add_argument("--baud", type=int, default=115200, help="Pixhawk serial baud rate (default: 115200).")
    parser.add_argument("--udp-port", type=int, default=14550, help="UDP broadcast port (default: 14550).")
    parser.add_argument("--tcp-port", type=int, default=5760, help="TCP stream port (default: 5760).")

    args = parser.parse_args()

    print("=" * 65)
    print(" ASV-X1 PIXHAWK USB MAVLINK ROUTER")
    print(f" Port     : {args.port}")
    print(f" Baudrate : {args.baud}")
    print(f" UDP Port : {args.udp_port}")
    print(f" TCP Port : {args.tcp_port}")
    print("=" * 65)

    router = MavlinkRouter(port=args.port, baudrate=args.baud, udp_port=args.udp_port, tcp_port=args.tcp_port)
    if not router.start():
        sys.exit(1)

    print("[INFO] Router running. Press Ctrl+C to terminate.\n")
    try:
        while True:
            time.sleep(2.0)
            total_net = (router.bytes_from_udp + router.bytes_from_tcp) // 1024
            print(
                f"\r[STATUS] Serial ➔ Net: {router.bytes_from_serial // 1024} KB | "
                f"Net ➔ Serial: {total_net} KB | "
                f"UDP: {len(router.known_udp_clients)} | "
                f"TCP: {len(router.active_tcp_clients)}",
                end="",
                flush=True
            )
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down MAVLink router...")
    finally:
        router.stop()


if __name__ == "__main__":
    main()

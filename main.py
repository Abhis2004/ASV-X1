"""
ASV-X1 Unified Deployment Entrypoint (Raspberry Pi 5 + Hailo-8L + Direct USB MAVLink)
Runs the autonomous drone tracking, live MJPEG streaming, and USB MAVLink router in a single command:

    python main.py
"""

import os
import sys
import argparse

# Ensure current directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import track


def main():
    parser = argparse.ArgumentParser(
        description="ASV-X1 Autonomous Drone Tracking & USB Telemetry System (Raspberry Pi 5 + Hailo-8L AI Hat)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: '0' for Raspberry Pi Camera (Picamera2 CSI ribbon) or USB webcam, or path to video file (default: '0')."
    )
    parser.add_argument(
        "--model", type=str, default="auto",
        help="Path to YOLO model (default: 'auto', detects model/best.hef for Hailo-8L 26 TOPS NPU acceleration)."
    )
    parser.add_argument(
        "--swap-rb", action="store_true",
        help="Manually invert Red and Blue channels if your camera sensor produces swapped colors."
    )
    parser.add_argument(
        "--no-letterbox", action="store_true",
        help="Disable aspect-ratio letterboxing."
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable live MJPEG video streaming server."
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080,
        help="HTTP MJPEG video streaming port (default: 8080)."
    )
    parser.add_argument(
        "--stream-quality", type=int, default=65,
        help="JPEG quality for streaming 1-100 (default: 65)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV GUI window."
    )
    parser.add_argument(
        "--mav-port", type=str, default="auto",
        help="Pixhawk USB serial port (default: 'auto', detects /dev/ttyACM* or /dev/serial/by-id/*)."
    )
    parser.add_argument(
        "--mav-baud", type=int, default=115200,
        help="Pixhawk USB baud rate (default: 115200)."
    )
    parser.add_argument(
        "--no-mav-router", action="store_true",
        help="Disable automatic Pixhawk USB MAVLink router."
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" ASV-X1 DRONE TRACKING & TELEMETRY SYSTEM")
    print(" Platform : Raspberry Pi 5 + Hailo-8L AI Hat (26 TOPS)")
    print(f" Video    : Source '{args.source}'")
    print(f" Model    : '{args.model}'")
    if not args.no_stream:
        print(f" Stream   : MJPEG Video broadcast on port {args.stream_port}")
    if not args.no_mav_router:
        print(f" MAVLink  : USB Serial ({args.mav_port} @ {args.mav_baud}) ➔ UDP 14550")
    print("=" * 65)

    router = None
    if not args.no_mav_router:
        try:
            from mavlink_router import MavlinkRouter
            router = MavlinkRouter(port=args.mav_port, baudrate=args.mav_baud, udp_port=14550)
            if router.start():
                print("[INFO] USB MAVLink router active. Pixhawk telemetry forwarding to UDP 14550.")
            else:
                print("[WARN] USB MAVLink router waiting for Pixhawk USB connection...")
        except Exception as exc:
            print(f"[WARN] Could not initialize USB MAVLink router: {exc}")

    print("\n[INFO] Launching tracking pipeline and live video feed...")
    if not args.headless:
        print("[INFO] Press 'q' in the video window to cleanly exit.\n")

    try:
        track(
            video_path=args.source,
            model_path=args.model,
            swap_rb=args.swap_rb,
            letterbox=not args.no_letterbox,
            stream=not args.no_stream,
            stream_port=args.stream_port,
            stream_quality=args.stream_quality,
            headless=args.headless,
        )
    finally:
        if router is not None:
            router.stop()


if __name__ == "__main__":
    main()

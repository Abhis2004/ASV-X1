# ASV-X1 Raspberry Pi 5 Deployment

This folder contains everything you need to run the drone tracking project on your Raspberry Pi 5 with the **Raspberry Pi AI Kit (Hailo-8L 26 TOPS)** and camera modules.

---

## 1. System & Package Setup

On **Raspberry Pi OS (64-bit Bookworm)**:

### A. Install Camera Support (`picamera2` & OpenCV)
On Raspberry Pi 5, CSI ribbon cameras (IMX477 / Camera Module 3) require `libcamera` / `picamera2` rather than legacy V4L2 OpenCV video capture:
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-scipy
```

### B. Install Python Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```
*(or within a virtual environment if you use one)*

### C. Install Hailo Software Stack (for AI Hat 26 TOPS)
To enable the Hailo-8L accelerator:
```bash
sudo apt install -y hailo-all
sudo reboot
```
Verify the AI Hat is detected:
```bash
hailortcli fw-control identify
```

---

## 2. Running Everything in 1 Command (`main.py`)

Simply run:
```bash
python main.py
```

This single command automatically:
1. Initializes your **Picamera2** (CSI camera) or USB camera.
2. Loads your **Hailo-8L 26 TOPS NPU accelerator** (`model/best.hef`).
3. Launches the full OpenCV tracking window with tactical lock-on reticle.
4. Broadcasts the live annotated video feed via an **HTTP MJPEG server on port 8080** for the Flutter Ground Station (`asv_ground`).
5. Transmits automatic UDP network discovery beacons on port 14555 for zero-config app pairing.

### Optional Arguments:
- **Custom Video File**:
  ```bash
  python main.py --source test_video.mp4
  ```
- **Custom USB Webcam**:
  ```bash
  python main.py --source /dev/video0
  ```
- **Change Video Stream Port (default: 8080)**:
  ```bash
  python main.py --stream-port 8080
  ```
- **Adjust Stream Compression (1-100, default: 65, optimized for Pi 5 Wi-Fi)**:
  ```bash
  python main.py --stream-quality 70
  ```
- **Disable Video Streaming**:
  ```bash
  python main.py --no-stream
  ```
- **Invert Red/Blue Channels (if camera sensor requires it)**:
  ```bash
  python main.py --swap-rb
  ```

---

## 3. Hardware Acceleration on Hailo-8L (26 TOPS)

`model/best.hef` is **already present** in your `model/` folder.

When you run:
```bash
python inference.py --source 0
```

`inference.py` will automatically:
1. Detect `model/best.hef` and `model/metadata.yaml`.
2. Activate your **Hailo-8L 26 TOPS AI Hat NPU**.
3. Display real-time telemetry (FPS counter, `[HAILO-8L]` acceleration tag, target tracker, lock-on reticle) directly on screen!

---

## 4. Pixhawk Direct USB MAVLink Telemetry HUD (UDP 14550 & TCP 5760)

Pixhawk telemetry is routed directly via a standard USB cable (`/dev/ttyACM0`) using `mavlink_router.py` in a non-blocking background thread. Telemetry is broadcast over UDP 14550 and streamed over TCP 5760 across the Pi 5's Wi-Fi hotspot (`asv-network`), and rendered as a persistent bottom overlay in the OpenCV window.

### Features:
- **Persistent Bottom HUD**: Shows MAVLink connection status, Flight Mode (`LOITER`, `AUTO`, `STABILIZE`, etc.), `ARMED`/`DISARMED` state, Battery Voltage & Remaining %, Relative Altitude (AGL), Groundspeed, GPS Fix Type & Satellite Count, and Vehicle Heading.
- **Zero FPS Impact**: Telemetry runs asynchronously in a daemon thread so OpenCV and Hailo-8L video processing run at maximum framerate.
- **Direct USB Keepalive Routing**: Periodically emits 1 Hz GCS heartbeats and requests MAVLink data streams (`ATTITUDE`, `VFR_HUD`, `GLOBAL_POSITION_INT`, `SYS_STATUS`, `GPS_RAW_INT`) to keep the Pixhawk USB stream active.

### Standalone Telemetry Testing (Terminal Monitor):
To verify telemetry reception directly in the console without opening OpenCV:
```bash
python mavlink_router.py
```
Or run the unified deployment:
```bash
python main.py
```

### Running with Tracking & Telemetry:
```bash
python inference.py --source 0
```
If your telemetry is on another UDP port or IP:
```bash
python inference.py --source 0 --mavlink udpin:0.0.0.0:14550
```



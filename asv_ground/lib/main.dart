import 'dart:async';
import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';
import 'dart:io';
import 'package:dart_mavlink/mavlink.dart';
import 'package:dart_mavlink/dialects/ardupilotmega.dart';
import 'package:flutter/services.dart';
import 'dart:math' as math;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:open_file/open_file.dart';
import 'mjpeg_view.dart';
// --- TEMPORARY ESP32 FEATURE START ---
import 'esp32_bridge.dart';
// --- TEMPORARY ESP32 FEATURE END ---

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeRight,
    DeviceOrientation.landscapeLeft,
  ]).then((_) {
    runApp(const MyApp());
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Drone Dashboard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  RawDatagramSocket? _socket;
  final MavlinkParser _udpParser = MavlinkParser(MavlinkDialectArdupilotmega());
  final MavlinkParser _tcpParser = MavlinkParser(MavlinkDialectArdupilotmega());

  double _pitch = 0.0;
  double _roll = 0.0;
  double _yaw = 0.0;
  
  double _altitude = 0.0;
  double _speed = 0.0;
  int _battery = -1;
  double _batteryVoltage = 0.0;
  int _packetCount = 0;
  String _socketError = "";
  DateTime _lastMavlinkTime = DateTime.now();
  Timer? _telemetryWatchdogTimer;
  
  int _satCount = 0;
  double _hdop = 0.0;
  String _flightMode = "UNKNOWN";
  List<int> _rcChannels = List.filled(8, 0);
  String? _armingError;
  
  LatLng _position = const LatLng(0, 0);

  bool _showHorizon = true;
  bool _showQuickTelem = true;
  StateSetter? _fcLogsDialogState;
  bool _isLogging = false;
  File? _logFile;
  List<File> _savedLogs = [];
  
  // Dataflash log download state
  List<LogEntry> _fcLogs = [];
  bool _fetchingFcLogs = false;
  LogEntry? _downloadingLog;
  int _downloadOffset = 0;
  int _downloadTotal = 1;
  RandomAccessFile? _downloadRaf;
  Timer? _downloadTimer;
  int _targetSystem = 1;
  int _targetComponent = 1;

  // Video Streaming Configuration (Pi 5 Hotspot / Wi-Fi)
  String _droneHost = "10.42.0.1";
  int _dronePort = 8080;
  RawDatagramSocket? _beaconSocket;
  // --- TEMPORARY ESP32 FEATURE START ---
  bool _fetchFromEsp32 = false;
  // --- TEMPORARY ESP32 FEATURE END ---

  /// Fast probe to test if an IP is actively serving the MJPEG video stream
  Future<String?> _probeDroneIp() async {
    final candidates = [
      _droneHost,
      "10.42.0.1",
    ];
    for (final ip in candidates) {
      try {
        final client = HttpClient()..connectionTimeout = const Duration(milliseconds: 350);
        final req = await client.getUrl(Uri.parse("http://$ip:8080/video_feed"));
        final res = await req.close();
        client.close(force: true);
        if (res.statusCode == 200) {
          return ip;
        }
      } catch (_) {}
    }
    return null;
  }

  /// Listens on UDP port 14555 for the Pi 5's auto-discovery beacon
  void _startBeaconListener() async {
    try {
      _beaconSocket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 14555, reuseAddress: true);
      _beaconSocket?.broadcastEnabled = true;
      _beaconSocket?.listen((RawSocketEvent event) {
        if (event == RawSocketEvent.read) {
          final datagram = _beaconSocket?.receive();
          if (datagram != null) {
            final text = String.fromCharCodes(datagram.data);
            if (text.startsWith("ASV_PI5_STREAM")) {
              final detectedIp = datagram.address.address;
              if (_droneHost != detectedIp) {
                setState(() {
                  _droneHost = detectedIp;
                });
                debugPrint("[AUTO-DETECT] Discovered Raspberry Pi 5 at: $_droneHost");
              }
            }
          }
        }
      });
    } catch (e) {
      debugPrint("Beacon listener bind error: $e");
    }
  }

  void _showDroneIpDialog() {
    final controller = TextEditingController(text: "$_droneHost:$_dronePort");
    bool isScanning = false;
    String scanResult = "";

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return Dialog(
            backgroundColor: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            insetPadding: const EdgeInsets.symmetric(horizontal: 32, vertical: 8),
            child: Container(
              constraints: const BoxConstraints(maxWidth: 540),
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(18, 12, 18, 12),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Header
                    Row(
                      children: [
                        const Icon(Icons.wifi, color: Colors.blueAccent, size: 20),
                        const SizedBox(width: 8),
                        const Expanded(
                          child: Text(
                            "Drone Video Connection",
                            style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87),
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.close, size: 20, color: Colors.black54),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(),
                          onPressed: () => Navigator.pop(context),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      "Enter or auto-detect the Raspberry Pi 5 IP address on the ASV network (asv-network):",
                      style: TextStyle(fontSize: 11, color: Colors.black87),
                    ),
                    const SizedBox(height: 10),
                    // Landscape row: Input Field + Auto-Detect Button
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Expanded(
                          child: TextField(
                            controller: controller,
                            style: const TextStyle(fontSize: 13),
                            decoration: const InputDecoration(
                              labelText: "Drone IP:Port",
                              hintText: "10.42.0.1:8080",
                              isDense: true,
                              contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
                              border: OutlineInputBorder(),
                              prefixIcon: Icon(Icons.videocam, size: 18),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
                            visualDensity: VisualDensity.compact,
                          ),
                          icon: isScanning
                              ? const SizedBox(
                                  width: 14,
                                  height: 14,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Icon(Icons.search, size: 16),
                          label: Text(
                            isScanning ? "Scanning..." : "Auto-Detect",
                            style: const TextStyle(fontSize: 12),
                          ),
                          onPressed: isScanning
                              ? null
                              : () async {
                                  setDialogState(() {
                                    isScanning = true;
                                    scanResult = "Probing asv-network hotspot...";
                                  });
                                  final foundIp = await _probeDroneIp();
                                  setDialogState(() {
                                    isScanning = false;
                                    if (foundIp != null) {
                                      controller.text = "$foundIp:8080";
                                      scanResult = "Found Pi 5 at $foundIp!";
                                    } else {
                                      scanResult = "No active Pi stream found. Ensure main.py is running.";
                                    }
                                  });
                                },
                        ),
                      ],
                    ),
                    if (scanResult.isNotEmpty) ...[
                      const SizedBox(height: 4),
                      Text(
                        scanResult,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w500,
                          color: scanResult.contains("Found") ? Colors.green[700] : Colors.orange[800],
                        ),
                      ),
                    ],
                    const SizedBox(height: 8),
                    // Footer: Hotspot IP info + Action Buttons
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          "Hotspot: 10.42.0.1 | Beacon: 14555",
                          style: TextStyle(fontSize: 10, color: Colors.black54),
                        ),
                        Row(
                          children: [
                            TextButton(
                              style: TextButton.styleFrom(visualDensity: VisualDensity.compact),
                              onPressed: () => Navigator.pop(context),
                              child: const Text("Cancel"),
                            ),
                            const SizedBox(width: 6),
                            ElevatedButton(
                              style: ElevatedButton.styleFrom(visualDensity: VisualDensity.compact),
                              onPressed: () {
                                final text = controller.text.trim();
                                if (text.isNotEmpty) {
                                  final parts = text.split(":");
                                  setState(() {
                                    _droneHost = parts[0].trim();
                                    if (parts.length > 1) {
                                      _dronePort = int.tryParse(parts[1].trim()) ?? 8080;
                                    }
                                  });
                                }
                                Navigator.pop(context);
                              },
                              child: const Text("Save & Connect"),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  void _showFcLogsDialog() {
    showDialog(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (dialogContext, setDialogState) {
            _fcLogsDialogState = setDialogState;
            return Dialog(
              backgroundColor: const Color(0xFF242424),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              child: Container(
                width: 520,
                height: 380,
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Dialog Header
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.assignment, color: Colors.orangeAccent, size: 20),
                            SizedBox(width: 8),
                            Text(
                              'FC Dataflash Logs',
                              style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                        Row(
                          children: [
                            IconButton(
                              icon: _fetchingFcLogs
                                  ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blue),
                                    )
                                  : const Icon(Icons.refresh, color: Colors.blue),
                              tooltip: 'Refresh / Query Logs',
                              onPressed: _fetchingFcLogs ? null : _requestFcLogs,
                            ),
                            IconButton(
                              icon: const Icon(Icons.close, color: Colors.white54),
                              onPressed: () {
                                _fcLogsDialogState = null;
                                Navigator.pop(context);
                              },
                            ),
                          ],
                        ),
                      ],
                    ),
                    const Divider(color: Colors.white24),
                    const SizedBox(height: 6),

                    // Log list
                    Expanded(
                      child: _fcLogs.isEmpty
                          ? const Center(
                              child: Text(
                                'No logs loaded.\nTap the refresh icon to query Pixhawk.',
                                textAlign: TextAlign.center,
                                style: TextStyle(color: Colors.white54, fontSize: 13),
                              ),
                            )
                          : ListView.builder(
                              itemCount: _fcLogs.length,
                              itemBuilder: (context, index) {
                                final log = _fcLogs[index];
                                final isDownloading = _downloadingLog?.id == log.id;

                                return ListTile(
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                                  title: Text(
                                    'Log #${log.id}',
                                    style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w500),
                                  ),
                                  subtitle: isDownloading
                                      ? LinearProgressIndicator(
                                          value: _downloadTotal == 0 ? 0 : _downloadOffset / _downloadTotal,
                                          backgroundColor: Colors.white12,
                                          color: Colors.orangeAccent,
                                        )
                                      : Text(
                                          '${(log.size / (1024 * 1024)).toStringAsFixed(2)} MB',
                                          style: const TextStyle(color: Colors.white54, fontSize: 12),
                                        ),
                                  trailing: isDownloading
                                      ? IconButton(
                                          icon: const Icon(Icons.cancel, color: Colors.red),
                                          onPressed: _cancelFcLogDownload,
                                        )
                                      : PopupMenuButton<String>(
                                          icon: const Icon(Icons.more_vert, color: Colors.white70),
                                          onSelected: (value) async {
                                            final directory = await getApplicationDocumentsDirectory();
                                            final path = '${directory.path}/dataflash_log_${log.id}.bin';
                                            final file = File(path);

                                            if (value == 'download') {
                                              _startFcLogDownload(log);
                                            } else if (value == 'info') {
                                              final dateStr = log.timeUtc == 0
                                                  ? "Unknown (No GPS time)"
                                                  : DateTime.fromMillisecondsSinceEpoch(log.timeUtc * 1000, isUtc: true).toLocal().toString();
                                              final isDownloaded = await file.exists();
                                              if (mounted) {
                                                showDialog(
                                                  context: context,
                                                  builder: (context) => AlertDialog(
                                                    title: Text('Log #${log.id} Info'),
                                                    content: Text('Date: $dateStr\nSize: ${(log.size / (1024 * 1024)).toStringAsFixed(2)} MB\nStatus: ${isDownloaded ? "Downloaded" : "On Drone"}'),
                                                    actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
                                                  ),
                                                );
                                              }
                                            } else if (value == 'delete') {
                                              if (await file.exists()) {
                                                await file.delete();
                                                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Local file deleted.')));
                                                setDialogState(() {});
                                              } else {
                                                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Not downloaded yet.')));
                                              }
                                            } else if (value == 'share') {
                                              if (await file.exists()) {
                                                try {
                                                  await Share.shareXFiles([XFile(path)], text: 'Dataflash Log #${log.id}');
                                                } catch (e) {
                                                  if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed: $e')));
                                                }
                                              } else {
                                                if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please download first.')));
                                              }
                                            }
                                          },
                                          itemBuilder: (context) => [
                                            const PopupMenuItem(value: 'download', child: Text('Download')),
                                            const PopupMenuItem(value: 'info', child: Text('Info')),
                                            const PopupMenuItem(value: 'share', child: Text('Share')),
                                            const PopupMenuItem(value: 'delete', child: Text('Delete Local')),
                                          ],
                                        ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    ).then((_) {
      _fcLogsDialogState = null;
    });
  }

  @override
  void initState() {
    super.initState();
    _startListening();
    _startBeaconListener();
    _loadLogs();
    
    _udpParser.stream.listen(_handleMavlinkFrame);
    _tcpParser.stream.listen(_handleMavlinkFrame);
  }

  void _handleMavlinkFrame(MavlinkFrame frame) {
    _lastMavlinkTime = DateTime.now();
    if (frame.message is Attitude) {
      final attitude = frame.message as Attitude;
      setState(() {
        _pitch = attitude.pitch;
        _roll = attitude.roll;
        _yaw = attitude.yaw;
      });
    } else if (frame.message is AttitudeQuaternion) {
      final q = frame.message as AttitudeQuaternion;
      final q0 = q.q1, q1 = q.q2, q2 = q.q3, q3 = q.q4;
      final roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
      final pitch = math.asin((2 * (q0 * q2 - q3 * q1)).clamp(-1.0, 1.0));
      final yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));
      setState(() {
        _pitch = pitch;
        _roll = roll;
        _yaw = yaw;
      });
    } else if (frame.message is VfrHud) {
      final hud = frame.message as VfrHud;
      setState(() {
        _speed = hud.groundspeed;
        if (_altitude == 0.0 && hud.alt != 0.0) {
          _altitude = hud.alt;
        }
      });
    } else if (frame.message is GlobalPositionInt) {
      final pos = frame.message as GlobalPositionInt;
      setState(() {
        _position = LatLng(pos.lat / 1E7, pos.lon / 1E7);
        _altitude = pos.relativeAlt / 1000.0;
      });
    } else if (frame.message is SysStatus) {
      final sys = frame.message as SysStatus;
      setState(() {
        _battery = sys.batteryRemaining;
        if (sys.voltageBattery > 0 && sys.voltageBattery != 65535) {
          _batteryVoltage = sys.voltageBattery / 1000.0;
        }
      });
    } else if (frame.message is BatteryStatus) {
      final bat = frame.message as BatteryStatus;
      setState(() {
        if (bat.batteryRemaining >= 0) {
          _battery = bat.batteryRemaining;
        }
        if (bat.voltages.isNotEmpty && bat.voltages[0] > 0 && bat.voltages[0] != 65535) {
          _batteryVoltage = bat.voltages[0] / 1000.0;
        }
      });
    } else if (frame.message is GpsRawInt) {
      final gps = frame.message as GpsRawInt;
      setState(() {
        _satCount = gps.satellitesVisible;
        _hdop = gps.eph != 65535 ? gps.eph / 100.0 : 0.0;
      });
    } else if (frame.message is Heartbeat) {
      final hb = frame.message as Heartbeat;
      debugPrint("[HB] sys=${frame.systemId} comp=${frame.componentId} type=${hb.type} auto=${hb.autopilot} mode=${hb.customMode} base=${hb.baseMode}");
      
      // Flight modes belong to the vehicle autopilot (exclude pure GCS type=6)
      if (hb.type != 6) {
        final newMode = _getFlightMode(hb.customMode, hb.type);
        final isArmed = (hb.baseMode & 128) != 0;
        setState(() {
          _flightMode = newMode;
          _targetSystem = frame.systemId;
          _targetComponent = frame.componentId;
          if (isArmed) {
            _armingError = null; // Clear PreArm error when vehicle successfully arms
          }
        });
      }
    } else if (frame.message is Statustext) {
      final st = frame.message as Statustext;
      try {
        final nullIdx = st.text.indexOf(0);
        final rawCodes = nullIdx != -1 ? st.text.sublist(0, nullIdx) : st.text;
        final cleanCodes = rawCodes.map((c) => c & 0xFF).where((c) => c >= 32 && c <= 126).toList();
        final msg = String.fromCharCodes(cleanCodes).trim();
        debugPrint("[STATUSTEXT] $msg");
        if (mounted && msg.isNotEmpty) {
          // Check for PreArm checks, arming failures, or FC warnings/errors
          final lower = msg.toLowerCase();
          final isArmingOrError = lower.contains("prearm") ||
              lower.contains("arm") ||
              lower.contains("fail") ||
              lower.contains("error") ||
              lower.contains("bad") ||
              lower.contains("warn") ||
              lower.contains("switch") ||
              lower.contains("check");
          if (isArmingOrError || st.severity <= 4) {
            setState(() {
              _armingError = msg;
            });
          }
        }
      } catch (_) {}
    } else if (frame.message is RcChannels) {
      final rc = frame.message as RcChannels;
      setState(() {
        _rcChannels = [
          rc.chan1Raw, rc.chan2Raw, rc.chan3Raw, rc.chan4Raw,
          rc.chan5Raw, rc.chan6Raw, rc.chan7Raw, rc.chan8Raw,
        ];
      });
      debugPrint("[RC] CH1..8: ${_rcChannels.join(',')}");
    } else if (frame.message is RcChannelsRaw) {
      final rc = frame.message as RcChannelsRaw;
      setState(() {
        _rcChannels = [
          rc.chan1Raw, rc.chan2Raw, rc.chan3Raw, rc.chan4Raw,
          rc.chan5Raw, rc.chan6Raw, rc.chan7Raw, rc.chan8Raw,
        ];
      });
      debugPrint("[RC_RAW] CH1..8: ${_rcChannels.join(',')}");
    } else if (frame.message is LogEntry) {
      final log = frame.message as LogEntry;
      if (!_fcLogs.any((e) => e.id == log.id)) {
        setState(() {
          _fcLogs.add(log);
          _fcLogs.sort((a, b) => b.id.compareTo(a.id));
        });
        _fcLogsDialogState?.call(() {});
      }
    } else if (frame.message is LogData) {
      _handleLogData(frame.message as LogData);
    }
    
    _recordLog();
  }



  void _recordLog() {
    if (_isLogging && _logFile != null) {
      final line = "${DateTime.now().toIso8601String()},${_altitude.toStringAsFixed(2)},${_speed.toStringAsFixed(2)},$_battery,$_satCount,${_hdop.toStringAsFixed(2)},$_flightMode\n";
      _logFile!.writeAsStringSync(line, mode: FileMode.append);
    }
  }

  Future<void> _loadLogs() async {
    final directory = await getApplicationDocumentsDirectory();
    final files = directory.listSync().whereType<File>().where((file) => 
        file.path.endsWith('.csv') && (file.path != _logFile?.path)).toList();
    setState(() {
      _savedLogs = files;
    });
  }

  Future<void> _toggleLogging(bool value) async {
    if (value) {
      final directory = await getApplicationDocumentsDirectory();
      final path = '${directory.path}/telemetry_log_${DateTime.now().millisecondsSinceEpoch}.csv';
      _logFile = File(path);
      await _logFile!.writeAsString("Timestamp,Altitude(m),Speed(m/s),Battery(%),Satellites,HDOP,FlightMode\n");
      setState(() {
        _isLogging = true;
      });
      _loadLogs();
    } else {
      setState(() {
        _isLogging = false;
      });
      _logFile = null;
      _loadLogs();
    }
  }

  Socket? _tcpSocket;

  void _sendRawBytes(List<int> bytes) {
    if (_tcpSocket != null) {
      try {
        _tcpSocket!.add(bytes);
      } catch (_) {}
    }
    if (_socket != null) {
      // --- TEMPORARY ESP32 FEATURE START ---
      final targets = Esp32Bridge.getTargetIps(
        isEsp32Enabled: _fetchFromEsp32,
        normalDroneHost: _droneHost,
      );
      // --- TEMPORARY ESP32 FEATURE END ---
      for (final ip in targets) {
        try {
          _socket!.send(bytes, InternetAddress(ip), 14550);
        } catch (_) {}
      }
    }
  }

  void _sendFrame(MavlinkFrame frame) {
    _sendRawBytes(frame.serialize());
  }

  void _sendGcsHeartbeatAndWakeup() {
    // Valid MAVLink 1 GCS Heartbeat (sysid 255, compid 190, GCS active)
    final hbBytes = [254, 9, 0, 255, 190, 0, 0, 0, 0, 0, 6, 8, 0, 4, 3, 73, 33];
    // REQUEST_DATA_STREAM ALL @ 4 Hz
    final reqBytes = [254, 6, 0, 255, 190, 66, 4, 0, 1, 1, 0, 1, 151, 114];
    _sendRawBytes(hbBytes);
    _sendRawBytes(reqBytes);
  }

  void _requestFcLogs() {
    setState(() {
      _fcLogs.clear();
      _fetchingFcLogs = true;
    });
    _fcLogsDialogState?.call(() {});
    final frame = MavlinkFrame.v1(
      0, // sequence
      255, // systemId
      190, // componentId
      LogRequestList(targetSystem: _targetSystem, targetComponent: _targetComponent, start: 0, end: 0xFFFF),
    );
    _sendFrame(frame);
    
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        setState(() => _fetchingFcLogs = false);
        _fcLogsDialogState?.call(() {});
      }
    });
  }

  Future<void> _startFcLogDownload(LogEntry log) async {
    final directory = await getApplicationDocumentsDirectory();
    final path = '${directory.path}/dataflash_log_${log.id}.bin';
    final file = File(path);
    if (await file.exists()) {
      OpenFile.open(path);
      return;
    }
    
    _downloadRaf = await file.open(mode: FileMode.write);
    setState(() {
      _downloadingLog = log;
      _downloadTotal = log.size;
      _downloadOffset = 0;
    });
    _fcLogsDialogState?.call(() {});
    _requestLogDataChunk();
  }

  void _requestLogDataChunk() {
    if (_downloadingLog == null) return;
    final frame = MavlinkFrame.v1(
      0, // sequence
      255, // systemId
      190, // componentId
      LogRequestData(targetSystem: _targetSystem, targetComponent: _targetComponent, id: _downloadingLog!.id, ofs: _downloadOffset, count: 0xFFFFFFFF),
    );
    _sendFrame(frame);
    _resetDownloadTimer();
  }

  void _resetDownloadTimer() {
    _downloadTimer?.cancel();
    _downloadTimer = Timer(const Duration(milliseconds: 500), () {
      if (_downloadingLog != null) _requestLogDataChunk();
    });
  }

  void _handleLogData(LogData data) {
    if (_downloadingLog == null || _downloadRaf == null) return;
    if (data.id != _downloadingLog!.id) return;
    
    if (data.ofs == _downloadOffset) {
      _downloadRaf!.setPositionSync(data.ofs);
      _downloadRaf!.writeFromSync(data.data, 0, data.count);
      
      setState(() {
        _downloadOffset += data.count;
      });
      _fcLogsDialogState?.call(() {});
      _resetDownloadTimer();
      
      if (_downloadOffset >= _downloadTotal) {
        _finishFcLogDownload();
      }
    }
  }

  void _finishFcLogDownload() async {
    _downloadTimer?.cancel();
    await _downloadRaf?.close();
    _downloadRaf = null;
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Download Complete!')));
      setState(() {
        _downloadingLog = null;
      });
      _fcLogsDialogState?.call(() {});
    }
  }

  void _cancelFcLogDownload() async {
    _downloadTimer?.cancel();
    await _downloadRaf?.close();
    _downloadRaf = null;
    setState(() {
      _downloadingLog = null;
    });
    _fcLogsDialogState?.call(() {});
  }

  String _getFlightMode(int customMode, int vehicleType) {
    if (vehicleType == 2) { // Copter / Multi-rotor
      switch (customMode) {
        case 0: return 'STABILIZE';
        case 1: return 'ACRO';
        case 2: return 'ALT_HOLD';
        case 3: return 'AUTO';
        case 4: return 'GUIDED';
        case 5: return 'LOITER';
        case 6: return 'RTL';
        case 7: return 'CIRCLE';
        case 9: return 'LAND';
        case 11: return 'DRIFT';
        case 13: return 'SPORT';
        case 14: return 'FLIP';
        case 15: return 'AUTOTUNE';
        case 16: return 'POSHOLD';
        case 17: return 'BRAKE';
        case 18: return 'THROW';
        case 21: return 'SMART_RTL';
        default: return 'MODE $customMode';
      }
    } else if (vehicleType == 10 || vehicleType == 11) { // Rover / Boat / ASV
      switch (customMode) {
        case 0: return 'MANUAL';
        case 1: return 'ACRO';
        case 3: return 'STEERING';
        case 4: return 'HOLD';
        case 5: return 'LOITER';
        case 6: return 'FOLLOW';
        case 7: return 'SIMPLE';
        case 10: return 'AUTO';
        case 11: return 'RTL';
        case 12: return 'SMART_RTL';
        case 15: return 'GUIDED';
        default: return 'MODE $customMode';
      }
    } else if (vehicleType == 1) { // Fixed-Wing / Plane
      switch (customMode) {
        case 0: return 'MANUAL';
        case 1: return 'CIRCLE';
        case 2: return 'STABILIZE';
        case 3: return 'TRAINING';
        case 4: return 'ACRO';
        case 5: return 'FBWA';
        case 6: return 'FBWB';
        case 7: return 'CRUISE';
        case 8: return 'AUTOTUNE';
        case 10: return 'AUTO';
        case 11: return 'RTL';
        case 12: return 'LOITER';
        case 15: return 'GUIDED';
        default: return 'MODE $customMode';
      }
    }
    return 'MODE $customMode';
  }

  void _showFlightModeDialog() {
    final modes = [
      {"name": "STABILIZE", "mode": 0, "desc": "Self-leveling manual control (No GPS needed)"},
      {"name": "ALT_HOLD", "mode": 2, "desc": "Altitude hold via barometer (No GPS needed)"},
      {"name": "LOITER", "mode": 5, "desc": "GPS position hold (Requires 3D GPS lock)"},
      {"name": "POSHOLD", "mode": 16, "desc": "Natural position hold (Requires GPS)"},
      {"name": "RTL", "mode": 6, "desc": "Return To Launch (Requires GPS)"},
      {"name": "AUTO", "mode": 3, "desc": "Autonomous waypoint mission (Requires GPS)"},
      {"name": "ACRO", "mode": 1, "desc": "Rate-based manual acrobatics"},
      {"name": "GUIDED", "mode": 4, "desc": "Ground station commanded navigation"},
      {"name": "LAND", "mode": 9, "desc": "Automatic descent and disarm"},
    ];

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF242424),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.flight_takeoff, color: Colors.amberAccent, size: 20),
                SizedBox(width: 8),
                Text("Change Flight Mode", style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black26,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: Colors.white12),
              ),
              child: Text(
                "Current: $_flightMode | Sats: $_satCount (GPS ${_satCount >= 6 ? 'FIX OK' : 'NO FIX'})\nRC5 (Mode Ch): ${_rcChannels[4]} µs | RC6: ${_rcChannels[5]} µs",
                style: const TextStyle(color: Colors.white70, fontSize: 10, fontFamily: 'monospace'),
              ),
            ),
          ],
        ),
        content: SizedBox(
          width: 320,
          child: ListView.separated(
            shrinkWrap: true,
            itemCount: modes.length,
            separatorBuilder: (_, __) => const Divider(color: Colors.white12, height: 1),
            itemBuilder: (context, index) {
              final m = modes[index];
              final isCurrent = _flightMode == m["name"];
              return ListTile(
                dense: true,
                title: Text(
                  m["name"] as String,
                  style: TextStyle(
                    color: isCurrent ? Colors.amberAccent : Colors.white,
                    fontWeight: isCurrent ? FontWeight.bold : FontWeight.w500,
                  ),
                ),
                subtitle: Text(
                  m["desc"] as String,
                  style: const TextStyle(color: Colors.white54, fontSize: 10),
                ),
                trailing: isCurrent ? const Icon(Icons.check_circle, color: Colors.amberAccent, size: 16) : null,
                onTap: () {
                  Navigator.pop(context);
                  _setFlightMode(m["mode"] as int, m["name"] as String);
                },
              );
            },
          ),
        ),
      ),
    );
  }

  void _setFlightMode(int mode, String name) {
    debugPrint("[SET_MODE] Requesting mode $mode ($name)");
    // 1. Standard MAVLink SET_MODE message (ID 11)
    final setModeFrame = MavlinkFrame.v1(
      0,
      255, // GCS systemId
      190, // GCS componentId
      SetMode(
        targetSystem: _targetSystem,
        baseMode: 1, // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        customMode: mode,
      ),
    );
    _sendFrame(setModeFrame);

    // 2. Also send COMMAND_LONG with MAV_CMD_DO_SET_MODE (176) for maximum compatibility
    final cmdFrame = MavlinkFrame.v1(
      0,
      255,
      190,
      CommandLong(
        targetSystem: _targetSystem,
        targetComponent: _targetComponent,
        command: 176, // MAV_CMD_DO_SET_MODE
        confirmation: 0,
        param1: 1.0, // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        param2: mode.toDouble(),
        param3: 0.0,
        param4: 0.0,
        param5: 0.0,
        param6: 0.0,
        param7: 0.0,
      ),
    );
    _sendFrame(cmdFrame);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text("Requesting Flight Mode: $name..."),
        duration: const Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _reconnect() {
    setState(() {
      _packetCount = 0;
      _altitude = 0.0;
      _speed = 0.0;
      _battery = -1;
      _batteryVoltage = 0.0;
    });
    _recycleSockets();
  }

  void _bindUdpSocket() async {
    try {
      _socket?.close();
      _socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 14550, reuseAddress: true, reusePort: false);
      _socket?.broadcastEnabled = true;
      _socket?.listen((RawSocketEvent event) {
        if (event == RawSocketEvent.read) {
          final datagram = _socket?.receive();
          if (datagram != null && datagram.data.isNotEmpty) {
            _packetCount++;
            _udpParser.parse(datagram.data);
          }
        }
      }, onError: (err) {
        debugPrint("UDP socket event notice: $err");
      }, cancelOnError: false);
    } catch (e) {
      debugPrint("Socket binding error: $e");
    }
  }

  void _recycleSockets() {
    debugPrint("[WATCHDOG] Recycling telemetry network link...");
    _isConnectingTcp = false;
    try {
      _tcpSocket?.destroy();
    } catch (_) {}
    _tcpSocket = null;

    _bindUdpSocket();

    // --- TEMPORARY ESP32 FEATURE START ---
    if (_fetchFromEsp32) {
      Esp32Bridge.tryTcpConnect(_tryTcp);
    } else {
      _tryTcp(_droneHost);
      if (_droneHost != "10.42.0.1") {
        _tryTcp("10.42.0.1");
      }
    }
    // --- TEMPORARY ESP32 FEATURE END ---

    _sendGcsHeartbeatAndWakeup();
  }

  void _startListening() {
    _bindUdpSocket();

    _telemetryWatchdogTimer?.cancel();
    _telemetryWatchdogTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _sendGcsHeartbeatAndWakeup();

      // Telemetry Link Watchdog: if no MAVLink frames arrived for 4s, recycle stale sockets
      final secondsSinceLastFrame = DateTime.now().difference(_lastMavlinkTime).inSeconds;
      if (secondsSinceLastFrame >= 4) {
        _recycleSockets();
      } else if (_tcpSocket == null) {
        // --- TEMPORARY ESP32 FEATURE START ---
        if (_fetchFromEsp32) {
          Esp32Bridge.tryTcpConnect(_tryTcp);
        } else {
          _tryTcp(_droneHost);
          if (_droneHost != "10.42.0.1") {
            _tryTcp("10.42.0.1");
          }
        }
        // --- TEMPORARY ESP32 FEATURE END ---
      }
      // Smooth 1Hz UI refresh for packet counter & TCP status
      setState(() {});
    });

    // --- TEMPORARY ESP32 FEATURE START ---
    if (_fetchFromEsp32) {
      Esp32Bridge.tryTcpConnect(_tryTcp);
    } else {
      _tryTcp(_droneHost);
      if (_droneHost != "10.42.0.1") {
        _tryTcp("10.42.0.1");
      }
    }
    // --- TEMPORARY ESP32 FEATURE END ---
  }

  bool _isConnectingTcp = false;

  void _tryTcp(String ip, [int port = 5760]) async {
    if (_tcpSocket != null || _isConnectingTcp) return;
    _isConnectingTcp = true;
    try {
      final socket = await Socket.connect(ip, port, timeout: const Duration(seconds: 2));
      socket.setOption(SocketOption.tcpNoDelay, true);
      if (_tcpSocket != null) {
        socket.destroy();
        _isConnectingTcp = false;
        return;
      }
      if (mounted) {
        setState(() {
          _tcpSocket = socket;
        });
      } else {
        _tcpSocket = socket;
      }
      _isConnectingTcp = false;
      debugPrint("[TCP MAVLINK] Connected to $ip:$port (TCP_NODELAY enabled)");
      
      // Immediately send GCS heartbeat over TCP
      _sendGcsHeartbeatAndWakeup();

      socket.listen(
        (data) {
          _packetCount++;
          _tcpParser.parse(data);
        },
        onError: (_) {
          if (mounted) {
            setState(() => _tcpSocket = null);
          } else {
            _tcpSocket = null;
          }
        },
        onDone: () {
          if (mounted) {
            setState(() => _tcpSocket = null);
          } else {
            _tcpSocket = null;
          }
        },
        cancelOnError: true,
      );
    } catch (_) {
      _isConnectingTcp = false;
      // TCP not ready yet
    }
  }

  // --- TEMPORARY ESP32 FEATURE START ---
  void _setEsp32Mode(bool enable) {
    setState(() {
      _fetchFromEsp32 = enable;
      _packetCount = 0;
    });
    _tcpSocket?.destroy();
    _tcpSocket = null;

    if (enable) {
      debugPrint("[ESP32 MODE] Enabled - fetching telemetry from ESP32 hotspot");
      Esp32Bridge.tryTcpConnect(_tryTcp);
    } else {
      debugPrint("[ESP32 MODE] Disabled - normal mode (Pi 5)");
      _tryTcp(_droneHost);
      _tryTcp("10.42.0.1");
    }

    _sendGcsHeartbeatAndWakeup();

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          enable
              ? "Fetching data from ESP32 hotspot (192.168.2.1 / 192.168.4.1)"
              : "Returned to normal mode (Raspberry Pi 5)",
        ),
        backgroundColor: enable ? Colors.orange[800] : Colors.blueGrey[800],
        duration: const Duration(seconds: 3),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
  // --- TEMPORARY ESP32 FEATURE END ---

  @override
  void dispose() {
    _socket?.close();
    _tcpSocket?.destroy();
    _beaconSocket?.close();
    _downloadTimer?.cancel();
    _downloadRaf?.closeSync();
    _telemetryWatchdogTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final tcpStatus = _tcpSocket != null ? "TCP: OK" : "TCP: --";
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('DB Dashboard (Pkts: $_packetCount | $tcpStatus)'),
            // --- TEMPORARY ESP32 FEATURE START ---
            if (_fetchFromEsp32) ...[
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: Colors.orange[900],
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: Colors.orangeAccent, width: 1),
                ),
                child: const Text(
                  'ESP32',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ],
            // --- TEMPORARY ESP32 FEATURE END ---
          ],
        ),
        backgroundColor: const Color(0xFF1E1E1E),
        foregroundColor: Colors.white,
        actions: [
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert),
            tooltip: 'Menu',
            onSelected: (value) {
              // --- TEMPORARY ESP32 FEATURE START ---
              if (value == 'toggle_esp32') {
                _setEsp32Mode(!_fetchFromEsp32);
              } else
              // --- TEMPORARY ESP32 FEATURE END ---
              if (value == 'refresh') {
                _reconnect();
              } else if (value == 'fclogs') {
                _showFcLogsDialog();
              } else if (value == 'stream_settings') {
                _showDroneIpDialog();
              }
            },
            itemBuilder: (context) => [
              // --- TEMPORARY ESP32 FEATURE START ---
              Esp32Bridge.buildRadioMenuItem(
                context: context,
                isEnabled: _fetchFromEsp32,
              ),
              const PopupMenuDivider(),
              // --- TEMPORARY ESP32 FEATURE END ---
              const PopupMenuItem(
                value: 'refresh',
                child: Row(
                  children: [
                    Icon(Icons.refresh, color: Colors.blueAccent),
                    SizedBox(width: 10),
                    Text('Refresh Telemetry'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'fclogs',
                child: Row(
                  children: [
                    Icon(Icons.assignment, color: Colors.orangeAccent),
                    SizedBox(width: 10),
                    Text('FC Dataflash Logs'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'stream_settings',
                child: Row(
                  children: [
                    Icon(Icons.videocam, color: Colors.greenAccent),
                    SizedBox(width: 10),
                    Text('Stream Settings'),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: Stack(
        children: [
          // 1. Fullscreen Video Stream (takes 100% of the freed space)
          Positioned.fill(
            child: MjpegView(
              streamUrl: 'http://$_droneHost:$_dronePort/video_feed',
              onSettingsTap: _showDroneIpDialog,
            ),
          ),

          // 2. Small Bottom-Left Floating Artificial Horizon Popup / Gauge
          if (_showHorizon)
            Positioned(
              left: 14,
              bottom: 14,
              child: Container(
                width: 130,
                height: 130,
                clipBehavior: Clip.antiAlias,
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.8),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white38, width: 1.5),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black87,
                      blurRadius: 10,
                      offset: Offset(0, 4),
                    ),
                  ],
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: CustomPaint(
                        painter: ArtificialHorizonPainter(_pitch, _roll),
                      ),
                    ),
                    // Small close / minimize button
                    Positioned(
                      top: 4,
                      right: 4,
                      child: GestureDetector(
                        onTap: () => setState(() => _showHorizon = false),
                        child: Container(
                          padding: const EdgeInsets.all(3),
                          decoration: const BoxDecoration(
                            color: Colors.black54,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.close, color: Colors.white70, size: 13),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // Floating button to re-open artificial horizon if minimized
          if (!_showHorizon)
            Positioned(
              left: 14,
              bottom: 14,
              child: GestureDetector(
                onTap: () => setState(() => _showHorizon = true),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.75),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.white30, width: 1),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.speed, color: Colors.cyanAccent, size: 16),
                      SizedBox(width: 6),
                      Text(
                        "HUD",
                        style: TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
              ),
            ),

          // 3. Small Bottom-Right Floating Quick Telemetry Popup / Card
          if (_showQuickTelem)
            Positioned(
              right: 14,
              bottom: 14,
              child: Container(
                width: 165,
                height: 145,
                clipBehavior: Clip.antiAlias,
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.8),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white38, width: 1.5),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black87,
                      blurRadius: 10,
                      offset: Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    // Top header with title and close icon
                    Padding(
                      padding: const EdgeInsets.fromLTRB(10, 6, 6, 2),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Row(
                            children: [
                              Icon(Icons.analytics_outlined, color: Colors.orangeAccent, size: 13),
                              SizedBox(width: 5),
                              Text(
                                "QUICK TELEM",
                                style: TextStyle(
                                  color: Colors.white70,
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.8,
                                ),
                              ),
                            ],
                          ),
                          GestureDetector(
                            onTap: () => setState(() => _showQuickTelem = false),
                            child: Container(
                              padding: const EdgeInsets.all(3),
                              decoration: const BoxDecoration(
                                color: Colors.black54,
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(Icons.close, color: Colors.white70, size: 13),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Divider(color: Colors.white24, height: 4),
                    // 3x2 Metrics Grid (3 rows, 2 columns)
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        child: GridView.count(
                          crossAxisCount: 2,
                          childAspectRatio: 2.1,
                          physics: const NeverScrollableScrollPhysics(),
                          children: [
                            _buildMiniMetric("ALT", "${_altitude.toStringAsFixed(1)}m", Colors.purpleAccent),
                            _buildMiniMetric("SPD", "${_speed.toStringAsFixed(1)}m/s", Colors.orangeAccent),
                            _buildMiniMetric(
                              "BAT",
                              _battery >= 0
                                  ? "$_battery%"
                                  : (_batteryVoltage > 0 ? "${_batteryVoltage.toStringAsFixed(1)}V" : "--"),
                              _battery > 30 ? Colors.greenAccent : (_battery > 15 ? Colors.orangeAccent : Colors.redAccent),
                            ),
                            _buildMiniMetric("MODE", _flightMode, Colors.amberAccent, isSmall: true, onTap: _showFlightModeDialog),
                            _buildMiniMetric("SATS", "$_satCount", Colors.cyanAccent),
                            _buildMiniMetric("HDOP", _hdop.toStringAsFixed(1), Colors.blueAccent),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // Floating button to re-open Quick Telemetry if minimized
          if (!_showQuickTelem)
            Positioned(
              right: 14,
              bottom: 14,
              child: GestureDetector(
                onTap: () => setState(() => _showQuickTelem = true),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.75),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.white30, width: 1),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.analytics_outlined, color: Colors.orangeAccent, size: 16),
                      SizedBox(width: 6),
                      Text(
                        "TELEM",
                        style: TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
              ),
            ),

          // 4. Persistent Red FC Arming Error Banner at Bottom Center
          if (_armingError != null && _armingError!.isNotEmpty)
            Positioned(
              bottom: 12,
              left: _showHorizon ? 154 : 16,
              right: _showQuickTelem ? 234 : 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.red.shade900.withValues(alpha: 0.92),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.redAccent, width: 1.5),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black87,
                      blurRadius: 8,
                      offset: Offset(0, 3),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _armingError!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.5,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 6),
                    GestureDetector(
                      onTap: () => setState(() => _armingError = null),
                      child: Container(
                        padding: const EdgeInsets.all(2),
                        decoration: const BoxDecoration(
                          color: Colors.black38,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.close, color: Colors.white70, size: 14),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMiniMetric(String label, String value, Color color, {bool isSmall = false, VoidCallback? onTap}) {
    final content = Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white54, fontSize: 9, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 1),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: isSmall ? 10 : 12,
            fontWeight: FontWeight.bold,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );

    if (onTap != null) {
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onTap,
        child: content,
      );
    }
    return content;
  }
}

class ArtificialHorizonPainter extends CustomPainter {
  final double pitch;
  final double roll;

  ArtificialHorizonPainter(this.pitch, this.roll);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.clipRect(Rect.fromLTWH(0, 0, size.width, size.height));
    
    final center = Offset(size.width / 2, size.height / 2);
    
    canvas.translate(center.dx, center.dy);
    canvas.rotate(-roll);
    
    // Calculate pitch translation
    // Pitch is in radians, approximately +- PI/2. Let's scale it so that PI/2 = height/2
    final pitchOffset = (pitch / (math.pi / 2)) * (size.height / 2);
    canvas.translate(0, pitchOffset);
    
    // Draw sky
    final skyPaint = Paint()..color = Colors.blue;
    canvas.drawRect(Rect.fromLTRB(-size.width, -size.height, size.width, 0), skyPaint);
    
    // Draw ground
    final groundPaint = Paint()..color = Colors.brown;
    canvas.drawRect(Rect.fromLTRB(-size.width, 0, size.width, size.height), groundPaint);
    
    // Draw horizon line
    final horizonPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2.0;
    canvas.drawLine(Offset(-size.width, 0), Offset(size.width, 0), horizonPaint);
    
    // Restore transformations for fixed elements
    canvas.translate(0, -pitchOffset);
    canvas.rotate(roll);
    
    // Draw crosshair
    final crosshairPaint = Paint()
      ..color = Colors.red
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;
      
    // Center point
    canvas.drawCircle(Offset.zero, 3.0, Paint()..color = Colors.red);
    // Left wing
    canvas.drawLine(const Offset(-40, 0), const Offset(-15, 0), crosshairPaint);
    canvas.drawLine(const Offset(-15, 0), const Offset(-15, 10), crosshairPaint);
    // Right wing
    canvas.drawLine(const Offset(40, 0), const Offset(15, 0), crosshairPaint);
    canvas.drawLine(const Offset(15, 0), const Offset(15, 10), crosshairPaint);
  }

  @override
  bool shouldRepaint(covariant ArtificialHorizonPainter oldDelegate) {
    return oldDelegate.pitch != pitch || oldDelegate.roll != roll;
  }
}

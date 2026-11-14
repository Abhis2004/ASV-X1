import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// High-performance, live MJPEG video stream receiver with Click-To-Lock object tracking for Flutter.
/// Connects over Pi 5 Wi-Fi Hotspot (asv-network) to Raspberry Pi 5 MJPEG HTTP stream (:8080/video_feed)
/// and relays touchscreen taps to :8080/lock?nx=...&ny=... to lock/unlock tracked objects.
class MjpegView extends StatefulWidget {
  final String streamUrl;
  final BoxFit fit;
  final VoidCallback? onSettingsTap;
  final Function(bool isLocked, int? trackId)? onLockChanged;

  const MjpegView({
    super.key,
    required this.streamUrl,
    this.fit = BoxFit.contain,
    this.onSettingsTap,
    this.onLockChanged,
  });

  @override
  State<MjpegView> createState() => _MjpegViewState();
}

class _MjpegViewState extends State<MjpegView> with SingleTickerProviderStateMixin {
  Uint8List? _currentFrame;
  bool _isConnected = false;
  String _statusMessage = "Connecting to drone...";
  HttpClient? _client;
  StreamSubscription? _subscription;
  Timer? _reconnectTimer;
  Timer? _statusPollTimer;
  bool _disposed = false;

  // Aspect ratio extracted from MJPEG frame (defaults to 16:9)
  double _frameAspect = 16.0 / 9.0;

  // Target lock state
  bool _isTargetLocked = false;
  int? _lockedTrackId;
  String? _lockFeedbackMsg;
  Timer? _feedbackTimer;
  bool _isFollowing = false;
  String? _followStatus;
  double _vx = 0.0;
  double _yawRate = 0.0;

  // Touch reticle animation
  Offset? _tapPos;
  late AnimationController _reticleAnimController;

  @override
  void initState() {
    super.initState();
    _reticleAnimController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 550),
    )..addStatusListener((status) {
        if (status == AnimationStatus.completed) {
          if (mounted) setState(() => _tapPos = null);
        }
      });

    _startStream();
    _startStatusPolling();
  }

  @override
  void didUpdateWidget(covariant MjpegView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.streamUrl != widget.streamUrl) {
      _stopStream();
      _startStream();
    }
  }

  void _extractFrameAspect(Uint8List bytes) {
    try {
      int i = 0;
      while (i < bytes.length - 8) {
        if (bytes[i] == 0xFF) {
          final marker = bytes[i + 1];
          // SOF0 (0xC0), SOF1 (0xC1), SOF2 (0xC2)
          if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
            final h = (bytes[i + 5] << 8) | bytes[i + 6];
            final w = (bytes[i + 7] << 8) | bytes[i + 8];
            if (w > 0 && h > 0) {
              final aspect = w / h;
              if ((aspect - _frameAspect).abs() > 0.05) {
                _frameAspect = aspect;
              }
            }
            break;
          }
        }
        i++;
      }
    } catch (_) {}
  }

  void _startStream() async {
    if (_disposed) return;
    _stopStream();

    setState(() {
      _isConnected = false;
      _statusMessage = "Connecting to ${widget.streamUrl}...";
    });

    try {
      _client = HttpClient()..connectionTimeout = const Duration(seconds: 4);
      final request = await _client!.getUrl(Uri.parse(widget.streamUrl));
      request.headers.set(HttpHeaders.cacheControlHeader, "no-cache");
      final response = await request.close();

      if (response.statusCode != 200) {
        throw Exception("HTTP ${response.statusCode}");
      }

      if (_disposed) return;
      setState(() {
        _isConnected = true;
        _statusMessage = "Connected";
      });

      // Buffer incoming byte chunks and slice full JPEG frames (0xFFD8 to 0xFFD9)
      final List<int> buffer = [];
      const int marker = 0xFF;
      const int soi = 0xD8; // Start Of Image
      const int eoi = 0xD9; // End Of Image

      _subscription = response.listen(
        (List<int> chunk) {
          buffer.addAll(chunk);

          // Find start of JPEG
          int startIndex = -1;
          for (int i = 0; i < buffer.length - 1; i++) {
            if (buffer[i] == marker && buffer[i + 1] == soi) {
              startIndex = i;
              break;
            }
          }

          if (startIndex != -1) {
            // Find end of JPEG
            int endIndex = -1;
            for (int i = startIndex + 2; i < buffer.length - 1; i++) {
              if (buffer[i] == marker && buffer[i + 1] == eoi) {
                endIndex = i + 2;
                break;
              }
            }

            if (endIndex != -1) {
              final frameBytes = Uint8List.fromList(buffer.sublist(startIndex, endIndex));
              buffer.removeRange(0, endIndex);

              _extractFrameAspect(frameBytes);

              if (mounted) {
                setState(() {
                  _currentFrame = frameBytes;
                });
              }
            }
          }

          // Buffer safety limit (1 MB max)
          if (buffer.length > 1024 * 1024) {
            buffer.clear();
          }
        },
        onError: (err) {
          _handleDisconnect("Stream error: $err");
        },
        onDone: () {
          _handleDisconnect("Stream closed by drone");
        },
        cancelOnError: true,
      );
    } catch (e) {
      _handleDisconnect("Cannot reach drone stream");
    }
  }

  void _handleDisconnect(String reason) {
    if (_disposed) return;
    setState(() {
      _isConnected = false;
      _statusMessage = reason;
    });
    _subscription?.cancel();
    _client?.close(force: true);

    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 2), () {
      if (!_disposed) _startStream();
    });
  }

  void _stopStream() {
    _reconnectTimer?.cancel();
    _subscription?.cancel();
    _client?.close(force: true);
  }

  void _startStatusPolling() {
    _statusPollTimer?.cancel();
    _statusPollTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!mounted || _disposed || !_isConnected) return;
      _pollLockStatus();
    });
  }

  void _pollLockStatus() async {
    HttpClient? client;
    try {
      final uri = Uri.parse(widget.streamUrl);
      final host = uri.host.isNotEmpty ? uri.host : "10.42.0.1";
      final port = uri.port > 0 ? uri.port : 8080;
      final statusUri = Uri(
        scheme: uri.scheme.isNotEmpty ? uri.scheme : "http",
        host: host,
        port: port,
        path: "/status",
      );

      client = HttpClient()..connectionTimeout = const Duration(milliseconds: 1200);
      final req = await client.getUrl(statusUri).timeout(const Duration(milliseconds: 1200));
      req.headers.set(HttpHeaders.connectionHeader, "close");
      req.headers.set(HttpHeaders.cacheControlHeader, "no-cache");
      final resp = await req.close().timeout(const Duration(milliseconds: 1200));
      if (resp.statusCode == 200) {
        final body = await resp.transform(utf8.decoder).join().timeout(const Duration(milliseconds: 1200));
        final data = jsonDecode(body);
        if (data is Map<String, dynamic> && mounted) {
          final isLocked = data['locked'] == true;
          final tid = data['selected_id'] is int ? data['selected_id'] as int : null;
          final isFollowing = data['following'] == true;
          final followStatus = data['follow_status']?.toString();
          final vx = (data['vx'] is num) ? (data['vx'] as num).toDouble() : 0.0;
          final yawRate = (data['yaw_rate'] is num) ? (data['yaw_rate'] as num).toDouble() : 0.0;

          if (isLocked != _isTargetLocked ||
              tid != _lockedTrackId ||
              isFollowing != _isFollowing ||
              followStatus != _followStatus ||
              (vx - _vx).abs() > 0.05) {
            setState(() {
              _isTargetLocked = isLocked;
              _lockedTrackId = tid;
              _isFollowing = isFollowing;
              _followStatus = followStatus;
              _vx = vx;
              _yawRate = yawRate;
            });
            widget.onLockChanged?.call(isLocked, tid);
          }
        }
      }
    } catch (_) {
    } finally {
      client?.close(force: true);
    }
  }

  void _handleTap(TapDownDetails details, double renderW, double renderH, double offsetX, double offsetY) {
    final local = details.localPosition;
    final tapX = local.dx - offsetX;
    final tapY = local.dy - offsetY;

    // Check if tap fell inside the rendered video frame
    if (tapX < 0 || tapX > renderW || tapY < 0 || tapY > renderH) {
      return;
    }

    // Normalized coordinates (0.0 to 1.0)
    final nx = (tapX / renderW).clamp(0.0, 1.0);
    final ny = (tapY / renderH).clamp(0.0, 1.0);

    // Trigger visual reticle animation & tactile haptic pulse
    HapticFeedback.selectionClick();
    setState(() {
      _tapPos = local;
    });
    _reticleAnimController.forward(from: 0.0);

    // Send lock request to drone
    _sendLockRequest(nx, ny);
  }

  void _sendLockRequest(double nx, double ny) async {
    HttpClient? client;
    try {
      final uri = Uri.parse(widget.streamUrl);
      final host = uri.host.isNotEmpty ? uri.host : "10.42.0.1";
      final port = uri.port > 0 ? uri.port : 8080;
      final lockUri = Uri(
        scheme: uri.scheme.isNotEmpty ? uri.scheme : "http",
        host: host,
        port: port,
        path: "/lock",
        queryParameters: {
          "nx": nx.toStringAsFixed(4),
          "ny": ny.toStringAsFixed(4),
        },
      );

      debugPrint("[CLICK-TO-LOCK] 🎯 Sending tap coordinates: nx=${nx.toStringAsFixed(3)}, ny=${ny.toStringAsFixed(3)} -> $lockUri");

      client = HttpClient()..connectionTimeout = const Duration(milliseconds: 1500);
      final request = await client.getUrl(lockUri).timeout(const Duration(milliseconds: 1500));
      request.headers.set(HttpHeaders.connectionHeader, "close");
      request.headers.set(HttpHeaders.cacheControlHeader, "no-cache");
      final response = await request.close().timeout(const Duration(milliseconds: 1500));

      if (response.statusCode == 200) {
        final body = await response.transform(utf8.decoder).join().timeout(const Duration(milliseconds: 1500));
        debugPrint("[CLICK-TO-LOCK] ✅ Server response: $body");
        final data = jsonDecode(body);
        if (data is Map<String, dynamic> && mounted) {
          final isLocked = data['locked'] == true;
          final tid = data['selected_id'] is int ? data['selected_id'] as int : null;

          setState(() {
            _isTargetLocked = isLocked;
            _lockedTrackId = tid;
            _lockFeedbackMsg = isLocked ? "TARGET #$tid LOCKED" : (data['message']?.toString().toUpperCase() ?? "NO TARGET AT TAP");
          });

          widget.onLockChanged?.call(isLocked, tid);

          if (isLocked) {
            HapticFeedback.heavyImpact();
          } else {
            HapticFeedback.lightImpact();
          }

          _feedbackTimer?.cancel();
          _feedbackTimer = Timer(const Duration(seconds: 3), () {
            if (mounted) setState(() => _lockFeedbackMsg = null);
          });
        }
      } else {
        debugPrint("[CLICK-TO-LOCK] ⚠️ Server returned HTTP ${response.statusCode}");
      }
    } catch (e) {
      debugPrint("[CLICK-TO-LOCK] ❌ Error sending lock request: $e");
    } finally {
      client?.close(force: true);
    }
  }

  void _unlockTarget() async {
    HttpClient? client;
    try {
      final uri = Uri.parse(widget.streamUrl);
      final host = uri.host.isNotEmpty ? uri.host : "10.42.0.1";
      final port = uri.port > 0 ? uri.port : 8080;
      final unlockUri = Uri(
        scheme: uri.scheme.isNotEmpty ? uri.scheme : "http",
        host: host,
        port: port,
        path: "/unlock",
      );

      debugPrint("[CLICK-TO-LOCK] 🔓 Sending unlock request -> $unlockUri");
      client = HttpClient()..connectionTimeout = const Duration(milliseconds: 1500);
      final request = await client.getUrl(unlockUri).timeout(const Duration(milliseconds: 1500));
      request.headers.set(HttpHeaders.connectionHeader, "close");
      request.headers.set(HttpHeaders.cacheControlHeader, "no-cache");
      final response = await request.close().timeout(const Duration(milliseconds: 1500));

      if (response.statusCode == 200 && mounted) {
        setState(() {
          _isTargetLocked = false;
          _lockedTrackId = null;
          _isFollowing = false;
          _followStatus = "STANDBY";
          _vx = 0.0;
          _yawRate = 0.0;
          _lockFeedbackMsg = "TARGET UNLOCKED";
        });
        widget.onLockChanged?.call(false, null);
        HapticFeedback.lightImpact();

        _feedbackTimer?.cancel();
        _feedbackTimer = Timer(const Duration(seconds: 2), () {
          if (mounted) setState(() => _lockFeedbackMsg = null);
        });
      }
    } catch (e) {
      debugPrint("[CLICK-TO-LOCK] ❌ Error sending unlock: $e");
    } finally {
      client?.close(force: true);
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _reticleAnimController.dispose();
    _statusPollTimer?.cancel();
    _feedbackTimer?.cancel();
    _stopStream();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black,
      width: double.infinity,
      height: double.infinity,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final containerW = constraints.maxWidth;
          final containerH = constraints.maxHeight;

          // Compute exact letterboxed/pillarboxed video dimensions for BoxFit.contain
          double renderW, renderH, offsetX, offsetY;
          if (containerW / containerH > _frameAspect) {
            renderH = containerH;
            renderW = containerH * _frameAspect;
            offsetX = (containerW - renderW) / 2.0;
            offsetY = 0.0;
          } else {
            renderW = containerW;
            renderH = containerW / _frameAspect;
            offsetX = 0.0;
            offsetY = (containerH - renderH) / 2.0;
          }

          return GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTapDown: (details) => _handleTap(details, renderW, renderH, offsetX, offsetY),
            child: Stack(
              fit: StackFit.expand,
              children: [
                // 1. Live Video Frame accurately centered
                if (_currentFrame != null)
                  Center(
                    child: SizedBox(
                      width: renderW,
                      height: renderH,
                      child: Image.memory(
                        _currentFrame!,
                        fit: BoxFit.fill,
                        gaplessPlayback: true,
                      ),
                    ),
                  ),

                // 2. Animated Tap Reticle Ping
                if (_tapPos != null)
                  Positioned(
                    left: _tapPos!.dx - 30,
                    top: _tapPos!.dy - 30,
                    child: IgnorePointer(
                      child: SizedBox(
                        width: 60,
                        height: 60,
                        child: AnimatedBuilder(
                          animation: _reticleAnimController,
                          builder: (context, child) {
                            return CustomPaint(
                              painter: TapReticlePainter(
                                progress: _reticleAnimController.value,
                                color: _isTargetLocked ? Colors.redAccent : Colors.cyanAccent,
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  ),

                // 3. Top-Center Floating Tactical Lock-On HUD Badge
                if (_isTargetLocked || _lockFeedbackMsg != null)
                  Positioned(
                    top: 10,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: Builder(
                        builder: (context) {
                          String badgeText;
                          if (_lockFeedbackMsg != null) {
                            badgeText = _lockFeedbackMsg!;
                          } else if (_isTargetLocked) {
                            if (_isFollowing) {
                              badgeText = "TARGET #$_lockedTrackId LOCKED • FOLLOWING ${_vx > 0.05 ? '(${_vx.toStringAsFixed(1)} m/s)' : ''}".trim();
                            } else if (_followStatus != null && _followStatus!.contains("HOLDING")) {
                              badgeText = "TARGET #$_lockedTrackId LOCKED • HOLDING HOVER";
                            } else {
                              badgeText = "LOCKED ON TARGET #$_lockedTrackId";
                            }
                          } else {
                            badgeText = "TARGET UNLOCKED";
                          }

                          final Color badgeAccent = _isTargetLocked
                              ? (_isFollowing ? const Color(0xFF00E676) : Colors.redAccent)
                              : Colors.cyanAccent;
                          final Color badgeBg = _isTargetLocked
                              ? (_isFollowing ? const Color(0xFF0D381E) : Colors.red.shade900)
                              : Colors.black87;

                          return Container(
                            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                            decoration: BoxDecoration(
                              color: badgeBg.withValues(alpha: 0.90),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: badgeAccent,
                                width: 1.5,
                              ),
                              boxShadow: const [
                                BoxShadow(
                                  color: Colors.black54,
                                  blurRadius: 8,
                                  offset: Offset(0, 2),
                                ),
                              ],
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  _isTargetLocked
                                      ? (_isFollowing ? Icons.flight_takeoff : Icons.gps_fixed)
                                      : Icons.lock_open,
                                  color: badgeAccent,
                                  size: 15,
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  badgeText,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.8,
                                  ),
                                ),
                                if (_isTargetLocked) ...[
                                  const SizedBox(width: 10),
                                  GestureDetector(
                                    onTap: _unlockTarget,
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                                      decoration: BoxDecoration(
                                        color: Colors.white24,
                                        borderRadius: BorderRadius.circular(10),
                                      ),
                                      child: const Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(Icons.close, color: Colors.white, size: 11),
                                          SizedBox(width: 3),
                                          Text(
                                            "UNLOCK",
                                            style: TextStyle(
                                              color: Colors.white,
                                              fontSize: 9,
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                  ),

                // 4. Disconnection or Loading Overlay
                if (!_isConnected)
                  Container(
                    color: _currentFrame == null ? Colors.black87 : Colors.black45,
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const SizedBox(
                            width: 28,
                            height: 28,
                            child: CircularProgressIndicator(
                              strokeWidth: 2.5,
                              color: Colors.cyanAccent,
                            ),
                          ),
                          const SizedBox(height: 14),
                          Text(
                            _statusMessage,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            widget.streamUrl,
                            style: const TextStyle(color: Colors.white38, fontSize: 11),
                          ),
                        ],
                      ),
                    ),
                  ),

                // 5. Top-right Settings / URL Badge
                if (widget.onSettingsTap != null)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: GestureDetector(
                      onTap: widget.onSettingsTap,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.65),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: Colors.white24, width: 1),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              _isConnected ? Icons.videocam : Icons.videocam_off,
                              color: _isConnected ? Colors.greenAccent : Colors.orangeAccent,
                              size: 14,
                            ),
                            const SizedBox(width: 5),
                            Text(
                              widget.streamUrl.replaceAll("http://", "").replaceAll("/video_feed", ""),
                              style: const TextStyle(color: Colors.white70, fontSize: 11),
                            ),
                            const SizedBox(width: 4),
                            const Icon(Icons.edit, color: Colors.white38, size: 12),
                          ],
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}

/// Tactical reticle painter for touch feedback when clicking on objects to lock.
class TapReticlePainter extends CustomPainter {
  final double progress; // 0.0 to 1.0
  final Color color;

  TapReticlePainter({required this.progress, this.color = Colors.redAccent});

  @override
  void paint(Canvas canvas, Size size) {
    final opacity = (1.0 - progress).clamp(0.0, 1.0);
    final paint = Paint()
      ..color = color.withValues(alpha: opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = 10.0 + progress * 20.0;
    final cornerLen = 7.0;
    final rect = Rect.fromCircle(center: center, radius: radius);

    // Top-left corner
    canvas.drawLine(Offset(rect.left, rect.top + cornerLen), Offset(rect.left, rect.top), paint);
    canvas.drawLine(Offset(rect.left, rect.top), Offset(rect.left + cornerLen, rect.top), paint);

    // Top-right corner
    canvas.drawLine(Offset(rect.right - cornerLen, rect.top), Offset(rect.right, rect.top), paint);
    canvas.drawLine(Offset(rect.right, rect.top), Offset(rect.right, rect.top + cornerLen), paint);

    // Bottom-left corner
    canvas.drawLine(Offset(rect.left, rect.bottom - cornerLen), Offset(rect.left, rect.bottom), paint);
    canvas.drawLine(Offset(rect.left, rect.bottom), Offset(rect.left + cornerLen, rect.bottom), paint);

    // Bottom-right corner
    canvas.drawLine(Offset(rect.right - cornerLen, rect.bottom), Offset(rect.right, rect.bottom), paint);
    canvas.drawLine(Offset(rect.right, rect.bottom), Offset(rect.right, rect.bottom - cornerLen), paint);

    // Center crosshair pip
    canvas.drawCircle(center, 2.2, Paint()..color = color.withValues(alpha: opacity));
  }

  @override
  bool shouldRepaint(covariant TapReticlePainter oldDelegate) => oldDelegate.progress != progress;
}

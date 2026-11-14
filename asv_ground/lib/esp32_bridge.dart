// ============================================================
// TEMPORARY MODULAR FEATURE: ESP32 Telemetry Bridge
// This module provides telemetry fetching support when the phone
// is connected to an ESP32 Wi-Fi Hotspot (e.g. DroneBridge / MAVLink bridge).
// Can be completely removed at any time without impacting normal operation.
// ============================================================

import 'package:flutter/material.dart';

class Esp32Bridge {
  /// Common ESP32 Hotspot Gateway IPs:
  /// - 192.168.2.1: DroneBridge for ESP32 default SoftAP IP
  /// - 192.168.4.1: ESP-IDF / Arduino default SoftAP IP
  /// - 192.168.1.1: Alternative router/AP mode IP
  /// - 255.255.255.255: Subnet broadcast for automatic receiver discovery
  static const List<String> defaultCandidateIps = [
    "192.168.2.1",
    "192.168.4.1",
    "192.168.1.1",
    "255.255.255.255",
  ];

  static const List<int> candidateTcpPorts = [
    5760,  // Standard MAVLink TCP port
    20001, // DroneBridge for ESP32 default TCP port
  ];

  /// Returns outgoing target IPs depending on whether ESP32 mode is enabled
  static List<String> getTargetIps({
    required bool isEsp32Enabled,
    required String normalDroneHost,
  }) {
    if (isEsp32Enabled) {
      return defaultCandidateIps;
    }
    return [normalDroneHost, "10.42.0.1"];
  }

  /// Attempts TCP connections across candidate ESP32 IPs and ports
  static void tryTcpConnect(void Function(String ip, [int port]) tryTcpFn) {
    for (final ip in ["192.168.2.1", "192.168.4.1"]) {
      for (final port in candidateTcpPorts) {
        tryTcpFn(ip, port);
      }
    }
  }

  /// Builds the Radio PopupMenuItem for the 3-dot menu
  static PopupMenuItem<String> buildRadioMenuItem({
    required BuildContext context,
    required bool isEnabled,
  }) {
    return PopupMenuItem<String>(
      value: 'toggle_esp32',
      child: Row(
        children: [
          IgnorePointer(
            child: Radio<bool>(
              value: true,
              groupValue: isEnabled,
              activeColor: Colors.orangeAccent,
              onChanged: (_) {},
            ),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Fetch data from ESP32',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Text(
                  isEnabled ? 'Active (ESP32 Hotspot)' : 'Off (Normal / Pi 5)',
                  style: TextStyle(
                    fontSize: 10,
                    color: isEnabled ? Colors.orangeAccent : Colors.white38,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

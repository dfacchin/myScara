#!/usr/bin/env python3
"""
MQTT Broker Viewer - Connect to a local MQTT broker and display all messages.

Usage:
    python mqtt_viewer.py [--host HOST] [--port PORT]

Subscribes to all topics (#) and prints every message received.
Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from datetime import datetime

import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[{timestamp()}] Connected to MQTT broker at {userdata['host']}:{userdata['port']}")
        print(f"[{timestamp()}] Subscribing to all topics (#)...")
        print("-" * 70)
        client.subscribe("#")
    else:
        print(f"[{timestamp()}] Connection failed: {reason_code}")


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
    except UnicodeDecodeError:
        payload = f"<binary {len(msg.payload)} bytes: {msg.payload.hex()[:60]}>"

    print(f"[{timestamp()}] Topic: {msg.topic}")
    print(f"  QoS: {msg.qos} | Retain: {msg.retain} | Payload: {payload}")
    print()


def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"[{timestamp()}] Disconnected (rc={reason_code})")


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main():
    parser = argparse.ArgumentParser(description="MQTT Broker Viewer")
    parser.add_argument("--host", default="localhost", help="Broker host (default: localhost)")
    parser.add_argument("--port", type=int, default=1883, help="Broker port (default: 1883)")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata={"host": args.host, "port": args.port})
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print(f"[{timestamp()}] Connecting to {args.host}:{args.port}...")
    try:
        client.connect(args.host, args.port, keepalive=60)
    except ConnectionRefusedError:
        print(f"Error: Cannot connect to MQTT broker at {args.host}:{args.port}")
        print("Make sure a broker (e.g. mosquitto) is running.")
        sys.exit(1)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Stopped by user.")
        client.disconnect()


if __name__ == "__main__":
    main()

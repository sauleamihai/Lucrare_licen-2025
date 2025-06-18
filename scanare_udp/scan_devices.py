#!/usr/bin/env python3
import time
import json
import socket
import logging
import netifaces
import subprocess
import os
import socket as pysock
from datetime import datetime

# ─── Firebase Admin Setup ────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

# point to your service account key:
cred = credentials.Certificate("/home/Mihai/aplicatielicenta-bf604-firebase-adminsdk-fbsvc-b9882d1fba.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "aplicatielicenta-bf604.firebasestorage.app"
})
bucket = fb_storage.bucket()

# ─── CONFIGURATION (unchanged) ────────────────────────────────────────────
ARP_PATH    = "/proc/net/arp"
IFACE       = "wlan0"
UDP_PORT    = 4210
INTERVAL    = 10  # seconds between broadcasts
LEASE_FILES = [
    "/var/lib/misc/dnsmasq.leases",
    "/var/lib/NetworkManager/dnsmasq.d/dnsmasq.leases",
]

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG
)

# ─── (your helper functions, unchanged) ──────────────────────────────────
def read_core_voltage():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_volts", "core"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split("=",1)[1].rstrip("V"))
    except Exception as e:
        logging.error(f"Core voltage read failed: {e}")
        return None

def read_core_current():
    try:
        lines = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()
        for line in lines:
            if line.strip().startswith("VDD_CORE_A"):
                return float(line.split("=",1)[1].rstrip("A"))
    except Exception as e:
        logging.error(f"Core current read failed: {e}")
    return None

def get_system():
    sys = {}
    try:
        raw = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        sys["cpu_temp_c"] = int(raw) / 1000.0
    except Exception as e:
        logging.error(f"Temp read failed: {e}")
        sys["cpu_temp_c"] = None

    volts = read_core_voltage()
    amps  = read_core_current()
    sys["core_volts"]   = volts
    sys["power_core_w"] = (volts * amps) if (volts is not None and amps is not None) else None
    return sys

def get_lease_map():
    lease_map = {}
    for fname in LEASE_FILES:
        if not os.path.isfile(fname):
            continue
        with open(fname) as lf:
            for line in lf:
                parts = line.split()
                if len(parts) >= 4:
                    lease_map[parts[1].lower()] = parts[3]
    return lease_map

def get_devices(lease_map):
    devices = []
    with open(ARP_PATH) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 6: continue
            ip, _, flags, mac, _, dev = parts[:6]
            if dev != IFACE or flags == "0x0": continue
            name = lease_map.get(mac.lower(), "")
            if not name:
                try:
                    name = pysock.gethostbyaddr(ip)[0]
                except: name = ""
            devices.append({"ip": ip, "mac": mac, "name": name})
    return devices

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    addrs = netifaces.ifaddresses(IFACE).get(netifaces.AF_INET, [])
    bcast = addrs[0].get("broadcast","<broadcast>") if addrs else "<broadcast>"
    logging.info(f"Broadcasting to {bcast}:{UDP_PORT}")

    while True:
        lease_map   = get_lease_map()
        devices     = get_devices(lease_map)
        system_data = get_system()

        # 1) UDP broadcast (existing behavior)
        payload = json.dumps({
            "devices": devices,
            "system":  system_data
        })
        logging.debug(f"Payload: {payload}")
        try:
            sock.sendto(payload.encode(), (bcast, UDP_PORT))
            logging.info("Payload sent")
        except Exception as e:
            logging.error(f"UDP send error: {e}")

        # 2) Upload metrics to Firebase Storage
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **system_data
        }
        blob = bucket.blob(f"metrics/{record['timestamp']}.json")
        blob.upload_from_string(
            json.dumps(record),
            content_type="application/json"
        )
        logging.info(f"Uploaded metrics: {record}")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    logging.info("Starting raspi_device_scan")
    main()

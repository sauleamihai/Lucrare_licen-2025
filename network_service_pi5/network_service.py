#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# 1) Initialize the D-Bus main loop before any NetworkManager import
from dbus.mainloop.glib import DBusGMainLoop
DBusGMainLoop(set_as_default=True)

# 2) Now import NetworkManager safely
import NetworkManager
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

def get_device(ifname):
    """Retrieve the NM device object for a given interface name."""
    for dev in NetworkManager.NetworkManager.GetDevices():
        if dev.Interface == ifname:
            return dev
    abort(404, f"No such interface: {ifname}")

@app.route("/scan/<iface>", methods=["POST"])
def scan(iface):
    dev = get_device(iface)
    if not dev or dev.DeviceType != NetworkManager.NM_DEVICE_TYPE_WIFI:
        abort(404, "Wi-Fi interface not found")
    dev.RequestScan({})
    return jsonify({"status": "scanning", "interface": iface})

@app.route("/list/<iface>", methods=["GET"])
def list_networks(iface):
    dev = get_device(iface)
    if not dev or dev.DeviceType != NetworkManager.NM_DEVICE_TYPE_WIFI:
        abort(404, "Wi-Fi interface not found")
    aps = dev.SpecificDevice().GetAccessPoints()
    result = []
    for ap in aps:
        ssid = ap.Ssid.decode() if isinstance(ap.Ssid, bytes) else ap.Ssid
        result.append({
            "ssid": ssid,
            "strength": ap.Strength,
            "securityFlags": ap.Flags
        })
    return jsonify(result)

@app.route("/connect", methods=["POST"])
def connect():
    data = request.json or {}
    iface = data.get("iface")
    ssid  = data.get("ssid")
    psk   = data.get("psk")
    if not iface or not ssid:
        abort(400, "Missing 'iface' or 'ssid'")
    # never manage your hotspot interface
    if iface == "wlan0":
        abort(400, "Refusing to override hotspot on wlan0")

    conn_id = f"extender-{iface}"
    settings = {
        "connection": {
            "type": "802-11-wireless",
            "id": conn_id,
            "interface-name": iface,
            "autoconnect": True
        },
        "802-11-wireless": {
            "ssid": ssid,
            "mode": "infrastructure"
        },
        "ipv4": {"method": "auto"},
        "ipv6": {"method": "ignore"}
    }
    if psk:
        settings["802-11-wireless-security"] = {
            "key-mgmt": "wpa-psk",
            "psk": psk
        }

    # Find existing connection
    existing = None
    for c in NetworkManager.Settings.ListConnections():
        cfg = c.GetSettings().get("connection", {})
        if cfg.get("id") == conn_id:
            existing = c
            break
    if existing:
        existing.Update(settings)
        con = existing
    else:
        con = NetworkManager.Settings.AddConnection(settings)

    NetworkManager.NetworkManager.ActivateConnection(con, get_device(iface), "/")
    return jsonify({"status": "connecting", "ssid": ssid, "iface": iface})

@app.route("/active", methods=["GET"])
def active():
    active_cons = NetworkManager.NetworkManager.ActiveConnections
    return jsonify([
        {"id": ac.Id, "iface": ac.Devices[0].Interface}
        for ac in active_cons if ac.Devices
    ])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)

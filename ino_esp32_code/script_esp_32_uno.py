import network
import socket
import time
from machine import Pin
import ure

# ------------------------------------------------
# OPTIONAL: Captive Portal for WiFi Provisioning
# ------------------------------------------------
USE_CAPTIVE_PORTAL = True

def start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid='ESP32_Config', password='12345678')
    print("Starting Access Point for provisioning...")
    while not ap.active():
        pass
    print("AP active with SSID:", ap.config('essid'))
    print("AP config:", ap.ifconfig())

def start_captive_portal():
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.bind(addr)
    s.listen(1)
    print('Captive portal listening on', addr)
    while True:
        cl, addr = s.accept()
        print('Client connected from', addr)
        request = cl.recv(1024).decode('utf-8')
        print('Request:', request)
        if 'GET /?ssid=' in request:
            match = ure.search(r'/\?ssid=([^&]*)&password=([^ ]*)', request)
            if match:
                ssid = match.group(1)
                password = match.group(2)
                cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n')
                cl.send("Credentials received. Connecting to WiFi...")
                cl.close()
                s.close()
                return ssid, password
        # Serve HTML form
        html = """<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body { margin:0; padding:20px; font-family:Arial; background:#f4f4f4; }
  .container { max-width:400px; margin:auto; background:#fff; padding:20px; border-radius:8px; }
  h1 { text-align:center; }
  label { display:block; margin:8px 0 4px; font-weight:bold; }
  input { width:100%; padding:8px; margin-bottom:12px; box-sizing:border-box; }
  input[type="submit"] { background:#007bff; color:#fff; border:none; cursor:pointer; }
  input[type="submit"]:hover { background:#0056b3; }
</style></head><body>
  <div class="container">
    <h1>ESP32 WiFi Setup</h1>
    <form action="/" method="get">
      <label for="ssid">SSID</label>
      <input type="text" name="ssid" id="ssid" placeholder="Your WiFi SSID">
      <label for="password">Password</label>
      <input type="password" name="password" id="password" placeholder="Your WiFi Password">
      <input type="submit" value="Submit">
    </form>
  </div>
</body></html>
"""
        cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
        cl.send(html)
        cl.close()

# ------------------------------------------------
# WiFi Credentials & Ports
# ------------------------------------------------
SSID           = "Mihai"
PASSWORD       = "ZaR_20022002"
LOCAL_PORT     = 12345        # TCP port on ESP32
ARDUINO_IP     = "10.42.0.19"
ARDUINO_PORT   = 23456

# ------------------------------------------------
# Helpers for Matrix Handling
# ------------------------------------------------
def parse_matrix(data_str):
    """
    Convert semicolon-separated string into a 2D list of ints.
    Expects 3 rows × 16 comma-separated values each.
    """
    rows = [r for r in data_str.strip().split(";") if r]
    return [list(map(int, row.split(","))) for row in rows]

def send_matrix_to_arduino(matrix, ip, port):
    """
    Build a packet: 0xFF + 48 data bytes + 0xFE, send over TCP,
    wait for one-byte ACK (0x01) or NACK.
    """
    buf = bytearray([0xFF])
    for row in matrix:
        for val in row:
            buf.append(val if 0 <= val < 256 else 0)
    buf.append(0xFE)

    print("Sending to Arduino:", list(buf))
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((ip, port))
        time.sleep(0.1)
        s.sendall(buf)
        time.sleep(0.1)
        ack = s.recv(1)
        s.close()
        print("Arduino ACK:", ack)
        return ack == b'\x01'
    except Exception as e:
        print("Error sending to Arduino:", e)
        return False

# ------------------------------------------------
# Wi-Fi Connection & TCP Server
# ------------------------------------------------
def connect_to_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("Connecting to", ssid, "...")
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            time.sleep(1)
    ip = wlan.ifconfig()[0]
    print("Connected. IP address:", ip)
    return ip

def setup_server(ip, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((ip, port))
    srv.listen(1)
    print(f"Server listening on {ip}:{port}")
    return srv

# ------------------------------------------------
# Main Program
# ------------------------------------------------
def main():
    if USE_CAPTIVE_PORTAL:
        start_ap()
        ssid, password = start_captive_portal()
        network.WLAN(network.AP_IF).active(False)
        ip = connect_to_wifi(ssid, password)
    else:
        ip = connect_to_wifi(SSID, PASSWORD)

    server = setup_server(ip, LOCAL_PORT)
    conn, addr = server.accept()
    print("Client connected from", addr)

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            data_str = data.decode().strip()
            print("Received:", data_str)
            matrix = parse_matrix(data_str)
            for row in matrix:
                print(row)
            ok = send_matrix_to_arduino(matrix, ARDUINO_IP, ARDUINO_PORT)
            conn.sendall(b"ACK" if ok else b"NACK")
            print("Sent", "ACK" if ok else "NACK") 
            time.sleep(0.1)
    finally:
        conn.close()
        server.close()

if __name__ == "__main__":
    main()


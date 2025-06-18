#!/usr/bin/env python3
from gpiozero import Button, RotaryEncoder
import subprocess
import time

# GPIO buttons setup
button_setup      = Button(17, bounce_time=0.2)
button_connect    = Button(27, bounce_time=0.2)
button_disconnect = Button(22, bounce_time=0.2)
button_check      = Button(24, bounce_time=0.2)
button_quit       = Button(23, bounce_time=0.2)

# Rotary encoder setup
encoder = RotaryEncoder(a=5, b=6, max_steps=1000)
prev_steps = encoder.steps

DEVICE_MAC = "FA:ED:41:8D:43:C6"  # Replace with your Bluetooth device's MAC address

# Bluetooth and audio functions
def run_bluetooth_setup():
    cmd = '''echo "power on\nagent on\ndefault-agent" | sudo bluetoothctl'''
    subprocess.run(cmd, shell=True)
    print("Bluetooth setup commands executed.")

def run_bluetooth_connect():
    cmd = f'''echo "pair {DEVICE_MAC}\ntrust {DEVICE_MAC}\nconnect {DEVICE_MAC}" | sudo bluetoothctl'''
    subprocess.run(cmd, shell=True)
    print("Bluetooth connect and trust commands executed.")
    # Announce via speaker
    subprocess.run(f'espeak "Bluetooth is connected" --stdout | paplay', shell=True)

def run_bluetooth_disconnect():
    cmd = f'''echo "disconnect {DEVICE_MAC}" | sudo bluetoothctl'''
    subprocess.run(cmd, shell=True)
    print("Bluetooth disconnect command executed.")

def run_check_sinks():
    result = subprocess.run("pactl list short sinks", shell=True, capture_output=True, text=True)
    print("PulseAudio sinks:")
    print(result.stdout)

def run_bluetooth_quit():
    cmd = '''echo "quit" | sudo bluetoothctl'''
    subprocess.run(cmd, shell=True)
    print("Bluetooth quit command executed.")

def adjust_volume(change):
    vol = change[1:] + ("+" if change.startswith("+") else "-")
    subprocess.run(f"amixer set Master {vol}", shell=True)
    print("Volume adjusted by", change)

print("Service running: monitoring buttons and rotary encoder")

while True:
    try:
        current_steps = encoder.steps
        if current_steps != prev_steps:
            adjust_volume("+5%" if current_steps > prev_steps else "-5%")
            prev_steps = current_steps

        if button_setup.is_pressed:
            run_bluetooth_setup()
            time.sleep(1)

        if button_connect.is_pressed:
            run_bluetooth_connect()
            time.sleep(1)

        if button_disconnect.is_pressed:
            run_bluetooth_disconnect()
            time.sleep(1)

        if button_check.is_pressed:
            run_check_sinks()
            time.sleep(1)

        if button_quit.is_pressed:
            run_bluetooth_quit()
            time.sleep(1)

        time.sleep(0.1)
    except Exception as e:
        print("Error in main loop:", e)
        time.sleep(1)


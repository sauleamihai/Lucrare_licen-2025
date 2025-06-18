#!/usr/bin/env python3
import time
import random
import subprocess
import serial
import sys
import logging
import threading
from datetime import datetime

from gpiozero import Button, LED
from smbus import SMBus
import pigpio
from adafruit_extended_bus import ExtendedI2C as I2C_ext
from geopy.geocoders import Nominatim
import mariadb

# -------------------- Logging Configuration --------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------- Global Variables --------------------
latest_gps_fix = None   # Will store the most recent (lat, lon)
gps_mode_active = False # Controls whether to poll GPS
gps_opened = False      # Indicates if GPIO4 is already opened

# -------------------- Initialize pigpio for GPS --------------------
pi = pigpio.pi()
if not pi.connected:
    logging.error("Could not connect to pigpio daemon!")
    sys.exit(1)
logging.info("GPS pigpio connection established.")
GPS_RX_PIN = 4
GPS_BAUD = 9600

# -------------------- Continuous GPS Reader Thread --------------------
def continuous_gps_reader(pi):
    global latest_gps_fix, gps_mode_active, gps_opened
    logging.info("Starting continuous GPS reader thread.")
    while True:
        if gps_mode_active:
            if not gps_opened:
                ret = pi.bb_serial_read_open(GPS_RX_PIN, GPS_BAUD)
                if ret != 0:
                    logging.error(f"Failed to open GPS serial on GPIO4, error code: {ret}")
                else:
                    gps_opened = True
                    logging.info("GPS serial connection opened on GPIO4.")
            if gps_opened:
                try:
                    (count, data) = pi.bb_serial_read(GPS_RX_PIN)
                    if count > 0 and data:
                        try:
                            gps_data = data.decode('utf-8', errors='ignore')
                            for line in gps_data.splitlines():
                                if line.startswith("$GPRMC"):
                                    parts = line.split(',')
                                    if len(parts) > 6 and parts[2] == 'A':  # Valid fix indicated by 'A'
                                        lat = convert_to_decimal(parts[3], parts[4])
                                        lon = convert_to_decimal(parts[5], parts[6])
                                        if lat is not None and lon is not None:
                                            latest_gps_fix = (lat, lon)
                                            logging.debug(f"GPS fix updated: lat={lat}, lon={lon}")
                        except Exception as e:
                            logging.error(f"Error decoding GPS data: {e}")
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Error in GPS reader loop: {e}")
                    time.sleep(1)
        else:
            if gps_opened:
                pi.bb_serial_read_close(GPS_RX_PIN)
                gps_opened = False
                logging.info("GPS serial connection closed as GPS mode is inactive.")
            time.sleep(1)

# Start the GPS reader thread as a daemon.
gps_thread = threading.Thread(target=continuous_gps_reader, args=(pi,), daemon=True)
gps_thread.start()

# -------------------- Database Functions --------------------
def get_db_connection():
    try:
        conn = mariadb.connect(
            user="db_user",            # Replace with your DB username
            password="your_password",  # Replace with your DB password
            host="localhost",
            port=3306,
            database="sensor_db"       # Replace with your database name
        )
        logging.debug("Database connection established.")
        return conn
    except mariadb.Error as e:
        logging.error(f"Error connecting to MariaDB: {e}")
        sys.exit(1)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode INT,
            voltage_hw FLOAT,
            soc_hw FLOAT,
            voltage_sw FLOAT,
            soc_sw FLOAT,
            gps_lat FLOAT,
            gps_lon FLOAT,
            relay1_state TINYINT,
            relay2_state TINYINT,
            username VARCHAR(255) DEFAULT NULL,
            password VARCHAR(255) DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()
    logging.info("Database initialized (table sensor_readings ensured).")

def log_sensor_data(mode, voltage_hw, soc_hw, voltage_sw, soc_sw, gps_lat, gps_lon, username_value=None, password_value=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        r1 = int(relay1.value)
        r2 = int(relay2.value)
        cursor.execute("""
            INSERT INTO sensor_readings 
            (mode, voltage_hw, soc_hw, voltage_sw, soc_sw, gps_lat, gps_lon, relay1_state, relay2_state, username, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (mode, voltage_hw, soc_hw, voltage_sw, soc_sw, gps_lat, gps_lon, r1, r2, username_value, password_value))
        conn.commit()
        conn.close()
        logging.info(f"{datetime.now()} - Sensor data logged (mode={mode}).")
    except mariadb.Error as e:
        logging.error(f"Error logging sensor data: {e}")

def purge_old_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sensor_readings WHERE timestamp < NOW() - INTERVAL 2 HOUR")
        conn.commit()
        conn.close()
        logging.info(f"{datetime.now()} - Old sensor data purged.")
    except mariadb.Error as e:
        logging.error(f"Error purging old data: {e}")

# Initialize the database
init_db()

# -------------------- PIN CONFIGURATION --------------------
main_button = Button(13, pull_up=True)
toggle_button = Button(19, pull_up=True)
relay1 = LED(16)
relay2 = LED(26)

# -------------------- UART CONFIGURATION --------------------
UART_PORT = '/dev/ttyS0'
UART_BAUDRATE = 9600
try:
    uart = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=1)
    logging.info("UART connection established.")
except Exception as e:
    logging.error(f"Error opening UART: {e}")
    sys.exit(1)

# -------------------- RANDOM CREDENTIAL GENERATION --------------------
CHAR_SET = "abcdefgijkl"
def generate_random_string(length, char_set):
    return ''.join(random.choice(char_set) for _ in range(length))
username = generate_random_string(4, CHAR_SET)
password = generate_random_string(12, CHAR_SET)
logging.info(f"Generated username: {username}")
logging.info(f"Generated password: {password}")

# -------------------- MODE MANAGEMENT --------------------
# Modes: 0 = username, 1 = password, 2 = battery, 3 = GPS
current_mode = 0

def toggle_mode():
    global current_mode, gps_mode_active
    current_mode = (current_mode + 1) % 4
    gps_mode_active = (current_mode == 3)
    if current_mode == 0:
        logging.info("Switched to USERNAME mode.")
        speak("Switched to username mode.", speed=175)
    elif current_mode == 1:
        logging.info("Switched to PASSWORD mode.")
        speak("Switched to password mode.", speed=175)
    elif current_mode == 2:
        logging.info("Switched to BATTERY mode.")
        speak("Switched to battery mode.", speed=175)
    elif current_mode == 3:
        logging.info("Switched to GPS mode.")
        speak("Switched to GPS mode.", speed=175)

# -------------------- ESPEAK TTS FUNCTION --------------------
def speak(text, speed=175):
    logging.debug(f"Speaking: {text}")
    subprocess.run(["espeak", "-s", str(speed), text])

# -------------------- BATTERY READING FUNCTIONS --------------------
MAX17043_ADDR = 0x36

def read_battery_values_hw():
    logging.debug("Reading hardware battery values...")
    bus = SMBus(1)
    voltage = None
    soc = None
    try:
        raw_v = bus.read_word_data(MAX17043_ADDR, 0x02)
        raw_v = ((raw_v & 0xFF) << 8) | (raw_v >> 8)
        voltage = (raw_v >> 4) * 1.25 / 1000.0
        logging.debug(f"Hardware voltage: {voltage}")
    except Exception as e:
        logging.error(f"Error reading hardware battery voltage: {e}")
    try:
        raw_soc = bus.read_word_data(MAX17043_ADDR, 0x04)
        raw_soc = ((raw_soc & 0xFF) << 8) | (raw_soc >> 8)
        soc = (raw_soc >> 8) + ((raw_soc & 0xFF) / 256.0)
        logging.debug(f"Hardware SOC: {soc}")
    except Exception as e:
        logging.error(f"Error reading hardware battery SOC: {e}")
    bus.close()
    return voltage, soc

def read_word_sw(i2c, reg):
    try:
        buf = bytearray(2)
        i2c.writeto_then_readfrom(MAX17043_ADDR, bytes([reg]), buf)
        return (buf[0] << 8) | buf[1]
    except Exception as e:
        logging.error(f"Software I2C error on reg {hex(reg)}: {e}")
        return None

def read_battery_values_sw():
    logging.debug("Reading software battery values...")
    try:
        i2c_sw = I2C_ext(8)
        voltage = None
        soc = None
        word_v = read_word_sw(i2c_sw, 0x02)
        if word_v is not None:
            voltage = (word_v >> 4) * 1.25 / 1000.0
            logging.debug(f"Software voltage: {voltage}")
        word_soc = read_word_sw(i2c_sw, 0x04)
        if word_soc is not None:
            soc = (word_soc >> 8) + ((word_soc & 0xFF) / 256.0)
            logging.debug(f"Software SOC: {soc}")
        return voltage, soc
    except Exception as e:
        logging.error(f"Error reading software battery sensor: {e}")
        return None, None

# -------------------- GPS READING FUNCTIONS --------------------
def convert_to_decimal(degree_str, direction):
    try:
        if direction in ['N', 'S']:
            degrees = int(degree_str[:2])
            minutes = float(degree_str[2:])
        else:
            degrees = int(degree_str[:3])
            minutes = float(degree_str[3:])
        dec = degrees + minutes / 60.0
        if direction in ['S', 'W']:
            dec = -dec
        return dec
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        return None

def get_location_details(lat, lon):
    logging.debug("Performing reverse geocoding...")
    geolocator = Nominatim(user_agent="service_app")
    try:
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'address' in location.raw:
            addr = location.raw['address']
            city = addr.get('city', addr.get('town', addr.get('village', '')))
            road = addr.get('road', '')
            suburb = addr.get('suburb', addr.get('neighbourhood', ''))
            details = []
            if road:
                details.append(road)
            if suburb:
                details.append(suburb)
            if city:
                details.append(city)
            message = " ".join(details)
            if message.strip() == "":
                message = location.address
            logging.debug(f"Location details: {message}")
            return message
        else:
            logging.debug("Location details not found.")
            return "Unknown location"
    except Exception as e:
        logging.error(f"Reverse geocoding error: {e}")
        return "Unknown location"

# -------------------- MODE TOGGLES & ANNOUNCEMENTS --------------------
def announce_current():
    if current_mode == 0:
        logging.info("Announcing USERNAME mode.")
        speak("Username is", speed=175)
        for letter in username:
            speak(letter, speed=100)
            time.sleep(0.5)
        log_sensor_data(0, None, None, None, None, None, None, username, None)
    elif current_mode == 1:
        logging.info("Announcing PASSWORD mode.")
        speak("Password is", speed=175)
        for letter in password:
            speak(letter, speed=100)
            time.sleep(0.5)
        log_sensor_data(1, None, None, None, None, None, None, None, password)
    elif current_mode == 2:
        logging.info("Announcing BATTERY information.")
        voltage_hw, soc_hw = read_battery_values_hw()
        voltage_sw, soc_sw = read_battery_values_sw()
        if soc_hw is not None and soc_sw is not None:
            battery_message = (f"Hardware battery level is {soc_hw:.1f} percent. "
                               f"Software battery level is {soc_sw:.1f} percent.")
            logging.info(battery_message)
            speak(battery_message, speed=175)
            log_sensor_data(2, voltage_hw, soc_hw, voltage_sw, soc_sw, None, None, None, None)
        else:
            speak("Error reading battery values.", speed=175)
    elif current_mode == 3:
        logging.info("Announcing GPS information.")
        if latest_gps_fix is not None:
            lat, lon = latest_gps_fix
            location_message = get_location_details(lat, lon)
            gps_message = f"GPS fix acquired. Location is {location_message}."
            logging.info(gps_message)
            speak(gps_message, speed=125)
            log_sensor_data(3, None, None, None, None, lat, lon, None, None)
            logging.debug(f"Current GPS fix: lat={lat}, lon={lon}")
        else:
            logging.info("No valid GPS fix available.")
            speak("No valid GPS fix available.", speed=175)
            log_sensor_data(3, None, None, None, None, None, None, None, None)

def toggle_mode():
    global current_mode, gps_mode_active
    current_mode = (current_mode + 1) % 4
    gps_mode_active = (current_mode == 3)
    if current_mode == 0:
        logging.info("Switched to USERNAME mode.")
        speak("Switched to username mode.", speed=175)
    elif current_mode == 1:
        logging.info("Switched to PASSWORD mode.")
        speak("Switched to password mode.", speed=175)
    elif current_mode == 2:
        logging.info("Switched to BATTERY mode.")
        speak("Switched to battery mode.", speed=175)
    elif current_mode == 3:
        logging.info("Switched to GPS mode.")
        speak("Switched to GPS mode.", speed=175)

# -------------------- BUTTON DEBOUNCE --------------------
def debounce(button, delay=0.2):
    time.sleep(delay)
    while button.is_pressed:
        time.sleep(0.01)

# -------------------- UART COMMAND PROCESSING --------------------
def process_uart_command(command):
    command = command.strip().upper()
    logging.info(f"Received UART command: {command}")
    if command == "ON1":
        relay1.on()
        logging.info("Relay 1 turned ON.")
        speak("Relay one turned on.", speed=175)
    elif command == "OFF1":
        relay1.off()
        logging.info("Relay 1 turned OFF.")
        speak("Relay one turned off.", speed=175)
    elif command == "ON2":
        relay2.on()
        logging.info("Relay 2 turned ON.")
        speak("Relay two turned on.", speed=175)
    elif command == "OFF2":
        relay2.off()
        logging.info("Relay 2 turned OFF.")
        speak("Relay two turned off.", speed=175)
    else:
        logging.warning("Unknown UART command.")
        speak("Unknown command.", speed=175)

# -------------------- MAIN LOOP & PERIODIC LOGGING --------------------
last_log_time = time.time()

def main():
    global last_log_time
    logging.info("Service started.")
    logging.info(f"Initial mode: {current_mode}")
    try:
        while True:
            if toggle_button.is_pressed:
                logging.debug("Toggle button pressed.")
                debounce(toggle_button)
                toggle_mode()
            if main_button.is_pressed:
                logging.debug("Main button pressed.")
                debounce(main_button)
                announce_current()
                logging.info("Announcement complete.")
            if uart.in_waiting > 0:
                line = uart.readline().decode('utf-8', errors='ignore')
                if line:
                    process_uart_command(line)
            current_time = time.time()
            if current_time - last_log_time >= 30:
                logging.info("Periodic logging triggered.")
                voltage_hw, soc_hw = read_battery_values_hw()
                voltage_sw, soc_sw = read_battery_values_sw()
                gps_fix = latest_gps_fix
                if gps_fix:
                    lat, lon = gps_fix
                else:
                    lat, lon = None, None
                # For periodic logging (mode code 4) we now always record username and password.
                log_sensor_data(4, voltage_hw, soc_hw, voltage_sw, soc_sw, lat, lon, username, password)
                purge_old_data()
                last_log_time = current_time
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Shutting down service...")
    finally:
        uart.close()

if __name__ == "__main__":
    main()

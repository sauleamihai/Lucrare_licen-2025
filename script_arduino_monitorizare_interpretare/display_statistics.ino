#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPI.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

// === DISPLAY PINS ===
#define TFT_CS   5
#define TFT_DC   2
#define TFT_RST  4

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// === WIFI / UDP ===
const char* SSID       = "Mihai";
const char* PASSWORD   = "ZaR_20022002";
IPAddress LOCAL_IP(10, 42, 0, 2);
IPAddress GATEWAY(10, 42, 0, 1);
IPAddress SUBNET(255, 255, 255, 0);
IPAddress DNS1(10, 42, 0, 1);
const uint16_t LISTEN_PORT = 4210;
WiFiUDP udp;

// === VERTICAL SHIFT ===
const int TOP_MARGIN = 3;

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n\n=== ESP32 Booting ===");

  // Initialize display
  Serial.println("Initializing TFT display...");
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  pinMode(15, OUTPUT);
  digitalWrite(15, HIGH);  // turn backlight on
  Serial.println("Display ready.");

  // Configure Wi-Fi
  Serial.printf("Setting WiFi mode to STA...\n");
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(100);

  Serial.printf("Configuring static IP: %s\n", LOCAL_IP.toString().c_str());
  bool ipOk = WiFi.config(LOCAL_IP, GATEWAY, SUBNET, DNS1);
  Serial.printf("  → WiFi.config() returned %s\n", ipOk ? "true" : "false");

  Serial.printf("Connecting to SSID '%s'...\n", SSID);
  WiFi.begin(SSID, PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    Serial.printf("  WiFi.status() = %d (attempt %d)\n", WiFi.status(), attempts+1);
    delay(500);
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf(" WiFi connected, IP = %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.printf(" WiFi connection failed, status = %d\n", WiFi.status());
  }

  // Start UDP listener
  udp.begin(LISTEN_PORT);
  Serial.printf("Listening for UDP on port %u\n", LISTEN_PORT);
}

void loop() {
  // Check for incoming UDP packets
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    Serial.printf("\n Packet received: %d bytes\n", packetSize);

    // Read the packet into buffer
    static char buf[512];
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len <= 0) {
      Serial.println(" Warning: zero-length packet");
      return;
    }
    buf[len] = '\0';
    Serial.print("Raw payload: ");
    Serial.println(buf);

    // Parse JSON
    StaticJsonDocument<1024> doc;
    DeserializationError err = deserializeJson(doc, buf);
    if (err) {
      Serial.print("JSON parse error: ");
      Serial.println(err.c_str());
      return;
    }
    Serial.println("JSON parsed successfully.");

    // Extract system metrics
    float temp  = doc["system"]["cpu_temp_c"]   | -1.0;
    float volts = doc["system"]["core_volts"]   | -1.0;
    float power = doc["system"]["power_core_w"] | -1.0;
    Serial.printf("System: temp=%.1f°C, volt=%.3fV, power=%.2fW\n", temp, volts, power);

    // Render on TFT
    Serial.println("Updating display...");
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);

    // Build header line
    char header[64];
    if (temp >= 0 && volts >= 0 && power >= 0) {
      snprintf(header, sizeof(header),
               "T:%.1fC V:%.3fV P:%.2fW",
               temp, volts, power);
    } else {
      strcpy(header, "Sys N/A");
    }

    // Center header
    int16_t x1,y1;
    uint16_t w,h;
    tft.getTextBounds(header, 0, 0, &x1, &y1, &w, &h);
    tft.setCursor((tft.width() - w) / 2, TOP_MARGIN);
    tft.print(header);

    // Draw device list
    tft.setCursor(0, TOP_MARGIN + 12);
    tft.print("IP           Host");
    Serial.println("Displaying device list:");
    int y = TOP_MARGIN + 22;
    for (JsonObject dev : doc["devices"].as<JsonArray>()) {
      if (y > tft.height() - 10) break;
      const char* ip   = dev["ip"];
      const char* name = dev["name"];
      tft.setCursor(0, y);
      tft.print(ip);
      Serial.printf("  %s", ip);
      if (name && *name) {
        tft.setCursor(70, y);
        tft.print(name);
        Serial.printf(" -> %s", name);
      }
      Serial.println();
      y += 10;
    }
    Serial.println("Display update complete.");
  }

  // Heartbeat if no packets
  static unsigned long last = 0;
  if (millis() - last > 2000) {
    Serial.print(".");
    last = millis();
  }

  delay(50);
}

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <HTTPClient.h>
#include <Base64.h>
#include <Preferences.h>
#include <ArduinoJson.h>

// ------------ CONFIGURARE ----------------

#define BUTTON_PIN       4    // buton pentru captură imagine

const char* apSSID     = "ESP32CAM_Config";
const char* apPassword = "config123";
const byte  DNS_PORT   = 53;

WebServer  server(80);
DNSServer  dnsServer;
Preferences preferences;

String serverURL = "http://fluskapi.ngrok.app/caption";

// Configurare limbă și prompt implicit
String defaultLanguage = "ro";  // Română ca limbă implicită
String defaultPrompt = "Descrie camera în care mă aflu, menționând obiectele și atmosfera din jur";

// Pinii ESP32-CAM AI-Thinker
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ------------ HTML + CSS + JS CONFIG ----------------
const char* config_html = R"rawliteral(
<!DOCTYPE HTML>
<html lang="ro">
<head>
  <meta charset="UTF-8">
  <title>ESP32-CAM Setup</title>
  <style>
    body {
      background: #f4f7f8;
      font-family: 'Arial', sans-serif;
      color: #333;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 20px;
      box-sizing: border-box;
    }
    .container {
      background: #fff;
      padding: 2rem 3rem;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      max-width: 500px;
      width: 100%;
    }
    h2 {
      margin-top: 0;
      text-align: center;
      color: #4CAF50;
    }
    label {
      display: block;
      margin: 1rem 0 0.5rem;
      font-weight: bold;
    }
    input[type="text"],
    input[type="password"],
    textarea,
    select {
      width: 100%;
      padding: 0.5rem;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-sizing: border-box;
      font-family: inherit;
    }
    textarea {
      min-height: 80px;
      resize: vertical;
    }
    .show-pass {
      margin-top: 0.5rem;
      display: flex;
      align-items: center;
      font-size: 0.9rem;
      color: #555;
    }
    .show-pass input {
      margin-right: 0.5rem;
      transform: scale(1.1);
    }
    input[type="submit"] {
      width: 100%;
      padding: 0.75rem;
      margin-top: 1.5rem;
      background: #4CAF50;
      border: none;
      border-radius: 4px;
      color: white;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    input[type="submit"]:hover {
      background: #45a049;
    }
    .note {
      font-size: 0.85rem;
      color: #777;
      text-align: center;
      margin-top: 1rem;
    }
    .section {
      margin-bottom: 1.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid #eee;
    }
    .section:last-child {
      border-bottom: none;
      margin-bottom: 0;
      padding-bottom: 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Configurează ESP32-CAM</h2>
    <form action="/save" method="POST">
      <div class="section">
        <h3>Setări WiFi</h3>
        <label for="ssid">SSID</label>
        <input type="text" id="ssid" name="ssid" placeholder="Nume rețea" required>

        <label for="password">Parolă</label>
        <input type="password" id="password" name="password" placeholder="Parolă rețea">

        <div class="show-pass">
          <input type="checkbox" id="showPass">
          <label for="showPass">Arată parola</label>
        </div>
      </div>

      <div class="section">
        <h3>Setări AI</h3>
        <label for="language">Limbă</label>
        <select id="language" name="language">
          <option value="ro" selected>Română</option>
          <option value="en">English</option>
          <option value="es">Español</option>
          <option value="fr">Français</option>
          <option value="de">Deutsch</option>
          <option value="it">Italiano</option>
        </select>

        <label for="prompt">Prompt personalizat</label>
        <textarea id="prompt" name="prompt" placeholder="Descrie camera în care mă aflu, menționând obiectele și atmosfera din jur">Descrie camera în care mă aflu, menționând obiectele și atmosfera din jur</textarea>
      </div>

      <input type="submit" value="Salvează">
    </form>
    <p class="note">Toate datele sunt stocate local pe dispozitiv.</p>
  </div>

  <script>
    document.getElementById('showPass').addEventListener('change', function(){
      const pwd = document.getElementById('password');
      pwd.type = this.checked ? 'text' : 'password';
    });
  </script>
</body>
</html>
)rawliteral";

// ------------ FUNCȚII WEB ----------------
void handleRoot() {
  server.send(200, "text/html", config_html);
}

void handleSave() {
  String newSSID = server.arg("ssid");
  String newPass = server.arg("password");
  String newLanguage = server.arg("language");
  String newPrompt = server.arg("prompt");
  
  if (newSSID.length() > 0) {
    preferences.begin("config", false);
    preferences.putString("ssid", newSSID);
    preferences.putString("pass", newPass);
    preferences.putString("language", newLanguage.length() > 0 ? newLanguage : defaultLanguage);
    preferences.putString("prompt", newPrompt.length() > 0 ? newPrompt : defaultPrompt);
    preferences.end();
    
    server.send(200, "text/html",
      "<div style='text-align:center;padding:50px;font-family:Arial;'>"
      "<h2 style='color:#4CAF50;'>Date salvate cu success!</h2>"
      "<p>Dispozitivul se va reporni în câteva secunde...</p>"
      "</div>");
    delay(3000);
    ESP.restart();
  } else {
    server.send(400, "text/html",
      "<div style='text-align:center;padding:50px;font-family:Arial;'>"
      "<h2 style='color:#e74c3c;'>Eroare: SSID invalid!</h2>"
      "<p><a href='/'>Încearcă din nou</a></p>"
      "</div>");
  }
}

void startConfigPortal() {
  Serial.println("Pornire Access Point pentru configurare...");
  WiFi.mode(WIFI_AP);
  WiFi.softAP(apSSID, apPassword);
  
  Serial.print("Access Point pornit. IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("Conectează-te la rețeaua: " + String(apSSID));
  Serial.println("Parola: " + String(apPassword));
  
  dnsServer.start(DNS_PORT, "*", WiFi.softAPIP());
  server.on("/", HTTP_GET, handleRoot);
  server.on("/save", HTTP_POST, handleSave);
  server.begin();
  
  while (true) {
    dnsServer.processNextRequest();
    server.handleClient();
    delay(10);
  }
}

// ------------ FUNCȚII CAMERA ----------------
void startCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    Serial.println("PSRAM găsit - folosesc setări înalte");
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    Serial.println("PSRAM nu este disponibil - folosesc setări standard");
  }
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Eroare inițializare cameră: 0x%x\n", err);
    return;
  }
  
  Serial.println("Camera inițializată cu succes!");
}

// ------------ FUNCȚIE PENTRU TRIMITEREA IMAGINII ----------------
void captureAndSendImage() {
  Serial.println("Buton apăsat - capturez imaginea...");
  
  // Capturează imaginea
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Eroare: Nu pot captura imaginea");
    return;
  }
  
  Serial.printf("Imagine capturată: %zu bytes\n", fb->len);
  
  // Encodez în base64
  String encoded = base64::encode(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  
  // Încarcă setările salvate
  preferences.begin("config", true);
  String language = preferences.getString("language", defaultLanguage);
  String prompt = preferences.getString("prompt", defaultPrompt);
  preferences.end();
  
  // Creez JSON-ul pentru request
  DynamicJsonDocument doc(encoded.length() + 1000);
  doc["image"] = encoded;
  doc["language"] = language;
  doc["prompt"] = prompt;
  doc["use_google_tts"] = true;
  
  String payload;
  serializeJson(doc, payload);
  
  Serial.println("Trimit cererea către server...");
  Serial.println("Limbă: " + language);
  Serial.println("Prompt: " + prompt);
  
  // Trimit request-ul HTTP
  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(30000); // 30 secunde timeout
  
  int httpResponseCode = http.POST(payload);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("Răspuns server (cod %d):\n", httpResponseCode);
    
    // Parsez răspunsul JSON pentru a afișa informații frumos formatate
    DynamicJsonDocument responseDoc(2048);
    DeserializationError error = deserializeJson(responseDoc, response);
    
    if (!error) {
      if (responseDoc.containsKey("response")) {
        Serial.println("=== DESCRIEREA CAMEREI ===");
        Serial.println(responseDoc["response"].as<String>());
        Serial.println("========================");
        if (responseDoc.containsKey("language_name")) {
          Serial.println("Limbă: " + responseDoc["language_name"].as<String>());
        }
        if (responseDoc.containsKey("tts_method")) {
          Serial.println("Metoda TTS: " + responseDoc["tts_method"].as<String>());
        }
      } else if (responseDoc.containsKey("error")) {
        Serial.println("Eroare de la server: " + responseDoc["error"].as<String>());
      }
    } else {
      Serial.println("Răspuns server (raw):");
      Serial.println(response);
    }
  } else {
    Serial.printf("Eroare HTTP: %d\n", httpResponseCode);
    Serial.println("Motivul erorii: " + http.errorToString(httpResponseCode));
  }
  
  http.end();
}

// ------------ SETUP & LOOP ----------------
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== ESP32-CAM AI Camera ===");
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Încarcă credențiale din NVS
  preferences.begin("config", true);
  String storedSSID = preferences.getString("ssid", "");
  String storedPass = preferences.getString("pass", "");
  String storedLanguage = preferences.getString("language", defaultLanguage);
  String storedPrompt = preferences.getString("prompt", defaultPrompt);
  preferences.end();

  Serial.println("Setări actuale:");
  Serial.println("- Limbă: " + storedLanguage);
  Serial.println("- Prompt: " + storedPrompt);

  // Dacă nu avem SSID salvat, deschidem portalul direct
  if (storedSSID == "") {
    Serial.println("Nu există credențiale WiFi salvate");
    startConfigPortal();
  }

  // Încearcă conectarea STA
  WiFi.mode(WIFI_STA);
  WiFi.begin(storedSSID.c_str(), storedPass.c_str());
  Serial.printf("Conectare la %s", storedSSID.c_str());

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 15000) {
    delay(500);
    Serial.print(".");
  }

  // Dacă timeout, trecem în configurare
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nConectare WiFi eșuată!");
    Serial.println("Pornesc modul de configurare...");
    startConfigPortal();
  }

  Serial.println("\n✓ WiFi conectat cu succes!");
  Serial.println("IP local: " + WiFi.localIP().toString());
  Serial.println("Putere semnal: " + String(WiFi.RSSI()) + " dBm");
  
  // Inițializez camera
  startCamera();
  
  Serial.println("\n=== Sistem gata! ===");
  Serial.println("Apasă butonul pentru a captura și descrie camera");
  Serial.println("Server: " + serverURL);
}

void loop() {
  // Verifică dacă WiFi este încă conectat
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi deconectat! Încerc reconectarea...");
    WiFi.reconnect();
    delay(5000);
    return;
  }
  
  // Captură imagine la apăsarea butonului
  static bool lastButtonState = HIGH;
  bool currentButtonState = digitalRead(BUTTON_PIN);
  
  // Detectez apăsarea butonului (trecerea de la HIGH la LOW)
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    delay(50); // Debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      captureAndSendImage();
      
      // Așteaptă până când butonul este eliberat pentru a evita capturile multiple
      while (digitalRead(BUTTON_PIN) == LOW) {
        delay(100);
      }
      delay(1000); // Pauză suplimentară pentru a evita apăsările accidentale
    }
  }
  
  lastButtonState = currentButtonState;
  delay(50);
}
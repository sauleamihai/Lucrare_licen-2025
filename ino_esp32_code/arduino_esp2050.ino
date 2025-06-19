#include "WiFiEsp.h"

// Emulate Serial1 on pins 6/7 if no hardware serial is available.
// If your board has a hardware Serial1, you can use that instead.
#ifndef HAVE_HWSERIAL1
#include "SoftwareSerial.h"
SoftwareSerial Serial1(6, 7); // RX, TX
#endif

// ----------------------
// WiFi Credentials
// ----------------------
char ssid[] = "Mihai";
char pass[] = "ZaR_20022002";
int status = WL_IDLE_STATUS;

// ----------------------
// WiFiEsp Server
// ----------------------
WiFiEspServer tcpServer(23456);

// ----------------------
// PCA9685 addresses
// Adjust as needed.
// ----------------------
#define PCA9685_ADDRESS   0x40
#define PCA9685_ADDRESS2  0x50
#define PCA9685_ADDRESS3  0x41

// ----------------------
// PCA9685 registers
// ----------------------
#define MODE1      0x00
#define MODE2      0x01
#define LED0_ON_L  0x06
#define LED0_OFF_L 0x08
#define PRESCALE   0xFE

// ----------------------
// Matrix & PCA9685 Setup
// ----------------------
#define ROWS 3
#define COLS 16
static uint8_t matrixData[ROWS][COLS];

static const int PACKET_SIZE = 50; // 1 start byte (0xFF) + 48 data bytes + 1 end byte (0xFE).

// ----------------------
// I2C (TWI) functions
// ----------------------

// For AVR (Uno/Mega), registers are used directly.
// Adjust if using a different MCU that doesn’t use these registers.
void i2c_init() {
  TWSR = 0;    // prescaler=1
  TWBR = 72;   // SCL ~100kHz at 16MHz => TWBR=72
  TWCR = (1 << TWEN);
}

void i2c_start() {
  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
  while (!(TWCR & (1 << TWINT))) { /* wait */ }
}

void i2c_stop() {
  TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
  _delay_ms(1);
}

void i2c_write(uint8_t data) {
  TWDR = data;
  TWCR = (1 << TWINT) | (1 << TWEN);
  while (!(TWCR & (1 << TWINT))) { /* wait */ }
}

void pca9685_write(uint8_t address, uint8_t reg, uint8_t value) {
  i2c_start();
  i2c_write(address << 1);
  i2c_write(reg);
  i2c_write(value);
  i2c_stop();
}

void pca9685_init(uint8_t address) {
  // Mode1, normal mode
  pca9685_write(address, MODE1, 0x00);
  _delay_ms(10);

  // Mode2, totem pole (OUTDRV)
  pca9685_write(address, MODE2, 0x04);

  // For ~50 Hz => prescale=0x79 (121 decimal)
  pca9685_write(address, PRESCALE, 0x79);
  _delay_ms(10);
}

// Set the ON and OFF counts for a specific channel.
// 'on' and 'off' are 12-bit values (0..4095).
void pca9685_set_pwm(uint8_t addr, uint8_t ch, uint16_t on, uint16_t off) {
  pca9685_write(addr, LED0_ON_L + 4 * ch,   on & 0xFF);
  pca9685_write(addr, LED0_ON_L + 4 * ch+1, on >> 8);
  pca9685_write(addr, LED0_OFF_L + 4 * ch,  off & 0xFF);
  pca9685_write(addr, LED0_OFF_L + 4 * ch+1, off >> 8);
}

// ----------------------
// Update motors/servos
// ----------------------
void update_motors_from_matrix() {
  uint8_t motor_index = 0; // 0..47 for 3×16
  for (uint8_t r = 0; r < ROWS; r++) {
    for (uint8_t c = 0; c < COLS; c++) {
      uint8_t intensity = matrixData[r][c]; // 0..255
      // Example: map 0..255 => 0..4000
      uint16_t pwm_val = map(intensity, 0, 255, 0, 4000);

      // Determine which PCA9685 board to use.
      uint8_t address;
      if (motor_index < 16) {
        address = PCA9685_ADDRESS;
      } else if (motor_index < 32) {
        address = PCA9685_ADDRESS2;
      } else {
        address = PCA9685_ADDRESS3;
      }

      // Channel is motor_index mod 16.
      uint8_t channel = motor_index % 16;

      // Set the PWM for that channel.
      pca9685_set_pwm(address, channel, 0, pwm_val);

      motor_index++;
    }
  }
}

// ----------------------
// setup()
// ----------------------
void setup() {
  Serial.begin(115200);

  // Init software serial or hardware serial for ESP module.
  Serial3.begin(115200);
  WiFi.init(&Serial3);

  // Check for the presence of the WiFi shield/module.
  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    while (true) { /* Halt */ }
  }

  // Attempt to connect to WiFi network.
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(1000);
  }
  Serial.println("You're connected to the network.");

  // Start the TCP server.
  tcpServer.begin();
  printWifiData();

  // Initialize I²C + PCA9685 boards
  i2c_init();
  pca9685_init(PCA9685_ADDRESS);
  pca9685_init(PCA9685_ADDRESS2);
  pca9685_init(PCA9685_ADDRESS3);

  // Optionally set all channels to 0 at startup
  for (uint8_t ch = 0; ch < 16; ch++) {
    pca9685_set_pwm(PCA9685_ADDRESS,  ch, 0, 0);
    pca9685_set_pwm(PCA9685_ADDRESS2, ch, 0, 0);
    pca9685_set_pwm(PCA9685_ADDRESS3, ch, 0, 0);
  }

  Serial.println("TCP Server + PCA9685 init complete. Waiting for clients...");
}

// ----------------------
// loop()
// ----------------------
void loop() {
  // Check if a client has connected.
  WiFiEspClient client = tcpServer.available();
  if (client) {
    Serial.println("Client connected. Waiting for data...");

    // We'll receive 50 bytes in total if the packet is correct:
    //  Start marker = 0xFF
    //  48 data bytes
    //  End marker   = 0xFE
    uint8_t buffer[PACKET_SIZE];
    int index = 0;
    const unsigned long TIMEOUT_MS = 3000;
    unsigned long startTime = millis();

    while (client.connected() && (millis() - startTime < TIMEOUT_MS) && index < PACKET_SIZE) {
      if (client.available()) {
        int c = client.read();
        if (c < 0) { // read error
          break;
        }
        buffer[index++] = (uint8_t)c;
      } else {
        delay(10); // small wait for data
      }
    }

    // Debug print the received bytes:
    Serial.print("Received bytes (");
    Serial.print(index);
    Serial.println("):");
    for (int i = 0; i < index; i++) {
      if (buffer[i] < 16) Serial.print('0');
      Serial.print(buffer[i], HEX);
      Serial.print(" ");
    }
    Serial.println();

    // Validate the packet
    bool validPacket = false;
    if (index == PACKET_SIZE && buffer[0] == 0xFF && buffer[PACKET_SIZE - 1] == 0xFE) {
      validPacket = true;
    }

    if (validPacket) {
      // Copy the 48 data bytes to matrixData
      int dataPos = 1; // position after 0xFF
      for (uint8_t r = 0; r < ROWS; r++) {
        for (uint8_t c = 0; c < COLS; c++) {
          matrixData[r][c] = buffer[dataPos++];
        }
      }

      // Debug: print the parsed matrix
      Serial.println("Matrix received (3×16):");
      for (uint8_t r = 0; r < ROWS; r++) {
        for (uint8_t c = 0; c < COLS; c++) {
          Serial.print(matrixData[r][c]);
          Serial.print(' ');
        }
        Serial.println();
      }

      // Update PCA9685 outputs
      update_motors_from_matrix();

      // Send back ACK=1
      client.write((uint8_t)0x01);
      Serial.println("Sent ACK=1 to client.");
    }
    else {
      Serial.println("Data format incorrect or timeout.");
      // Send NACK=0
      client.write((uint8_t)0x00);
    }

    delay(50); // ensure ACK/NACK is transmitted
    client.stop();
    Serial.println("Client disconnected.\n");
  }

  // Some delay to avoid tight looping
  delay(100);
}

// ----------------------
// Print WiFi Info
// ----------------------
void printWifiData() {
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  byte mac[6];
  WiFi.macAddress(mac);
  char buf[20];
  sprintf(buf, "%02X:%02X:%02X:%02X:%02X:%02X",
          mac[5], mac[4], mac[3], mac[2], mac[1], mac[0]);
  Serial.print("MAC address: ");
  Serial.println(buf);
}

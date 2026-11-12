// SentinelX ESP32 firmware
// - Receives camera alarms from the laptop over USB serial:
//     "FALL 1" / "FALL 0"   person fall detected / cleared
//     "ZONE 1" / "ZONE 0"   restricted area entered / cleared
//     "CLEAR"               switch off both camera alarms
// - Reads a DHT11 and runs the high-temperature alarm on its own
//   (works even with the laptop unplugged).
// - Every 2 s reports "STATE FALL 0 ZONE 0" (camera alarms) and
//   "TEMP 28.0 HUM 55 ALARM 0" (ALARM = high-temperature alarm only).
//
// Sound / light patterns (highest priority first):
//   FALL      fast urgent beeping (beep-beep-beep)   LED fast blink
//   HIGH TEMP two-tone siren                         LED slow blink
//   ZONE      steady buzz                            LED solid on
//
// Library: "DHT sensor library for ESPx" by beegee_tokyo

#include "DHTesp.h"

#define BUZZER_PIN    25
#define LED_PIN       26
#define DHT_PIN       27

#define TEMP_LIMIT    35.0   // °C: siren starts at or above this
#define TEMP_RESET    34.0   // °C: siren stops below this

#define FALL_HZ       3500   // fast beeps for a fall
#define FALL_STEP_MS  120
#define ZONE_HZ       2700   // steady buzz for restricted-area intrusion
#define SIREN_HIGH_HZ 3200   // two-tone siren for high temperature
#define SIREN_LOW_HZ  2200
#define SIREN_STEP_MS 300
#define DHT_READ_MS   2000
#define HOST_TIMEOUT_MS 5000 // laptop sends its state every second; if it stops, clear camera alarms

DHTesp dht;

String command = "";
bool fallAlarm = false;
bool zoneAlarm = false;
bool tempAlarm = false;
int currentHz = 0;
bool phase = false;
unsigned long lastStep = 0;
unsigned long lastDhtRead = 0;
unsigned long lastHostMsg = 0;

void setTone(int hz) {
  if (hz == currentHz) return;
  currentHz = hz;
  if (hz == 0) {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    tone(BUZZER_PIN, hz);
  }
}

void setFlag(bool &flag, bool value, const char *name) {
  if (flag != value) {
    flag = value;
    Serial.printf("%s %s\n", name, value ? "ON" : "OFF");
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  lastHostMsg = millis();
  if (cmd == "FALL 1")      setFlag(fallAlarm, true,  "FALL");
  else if (cmd == "FALL 0") setFlag(fallAlarm, false, "FALL");
  else if (cmd == "ZONE 1") setFlag(zoneAlarm, true,  "ZONE");
  else if (cmd == "ZONE 0") setFlag(zoneAlarm, false, "ZONE");
  else if (cmd == "CLEAR") {
    setFlag(fallAlarm, false, "FALL");
    setFlag(zoneAlarm, false, "ZONE");
  }
}

void readCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      handleCommand(command);
      command = "";
    } else {
      command += c;
      if (command.length() > 32) command = "";
    }
  }
  // Laptop crashed or cable pulled: don't leave the buzzer stuck on
  if ((fallAlarm || zoneAlarm) && millis() - lastHostMsg > HOST_TIMEOUT_MS) {
    Serial.println("HOST TIMEOUT - camera alarms cleared");
    fallAlarm = false;
    zoneAlarm = false;
  }
}

void readDht() {
  if (millis() - lastDhtRead < DHT_READ_MS) return;
  lastDhtRead = millis();

  // Alarm state report (sent even if the DHT11 fails, so the laptop can confirm the link)
  Serial.printf("STATE FALL %d ZONE %d\n", fallAlarm ? 1 : 0, zoneAlarm ? 1 : 0);

  TempAndHumidity v = dht.getTempAndHumidity();
  float t = v.temperature;
  float h = v.humidity;

  // DHT11 is rated 0-50 °C and 20-90 %RH; anything outside that is a bad read
  if (dht.getStatus() != DHTesp::ERROR_NONE || isnan(t) || isnan(h) ||
      t < 0 || t > 50 || h < 20 || h > 90) {
    Serial.printf("DHT ERROR %s (t=%.1f h=%.1f)\n", dht.getStatusString(), t, h);
    return;
  }

  if (!tempAlarm && t >= TEMP_LIMIT) tempAlarm = true;
  else if (tempAlarm && t < TEMP_RESET) tempAlarm = false;

  Serial.printf("TEMP %.1f HUM %.0f ALARM %d\n", t, h, tempAlarm ? 1 : 0);
}

bool stepEvery(unsigned long ms) {
  if (millis() - lastStep >= ms) {
    lastStep = millis();
    phase = !phase;
  }
  return phase;
}

void updateOutputs() {
  if (fallAlarm) {
    bool on = stepEvery(FALL_STEP_MS);
    setTone(on ? FALL_HZ : 0);
    digitalWrite(LED_PIN, on ? HIGH : LOW);
  } else if (tempAlarm) {
    bool high = stepEvery(SIREN_STEP_MS);
    setTone(high ? SIREN_HIGH_HZ : SIREN_LOW_HZ);
    digitalWrite(LED_PIN, high ? HIGH : LOW);
  } else if (zoneAlarm) {
    setTone(ZONE_HZ);
    digitalWrite(LED_PIN, HIGH);
  } else {
    setTone(0);
    digitalWrite(LED_PIN, LOW);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  dht.setup(DHT_PIN, DHTesp::DHT11);   // AUTO_DETECT misreads DHT11 as DHT22

  // Start-up self-test: short beep + LED flash. If you don't hear this, check the buzzer wiring.
  digitalWrite(LED_PIN, HIGH);
  tone(BUZZER_PIN, ZONE_HZ);
  delay(150);
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println("SentinelX ESP32 Ready");
}

void loop() {
  readCommands();
  readDht();
  updateOutputs();
}

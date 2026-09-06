// =============================================================================
// dual_ir.ino  -  PRIMARY DEPLOYMENT SKETCH
//
// Dual VL53L0X Time-of-Flight distance sensors + Adafruit #2167 IR beam-break
// sensor, running on an Arduino Uno. Streams ball-on-arc measurements to the
// host PC at 115200 baud over USB serial.
//
// Mode : SINGLE-SHOT ranging for both ToF sensors.
//        Continuous mode was tested (see archive/dual_ir_continuous/) but
//        is not used in deployment - cross-interference between the two
//        VL53L0X units facing each other on the arc produces unreliable
//        readings when they range simultaneously.
//
// Host side : balancer.hardware.create_distance_reader() parses this output.
//             Do NOT change the serial format without updating that parser.
//
// -----------------------------------------------------------------------------
// Pin map  (matches hardware/BOM_AND_ASSEMBLY.md §Arduino Wiring)
// -----------------------------------------------------------------------------
//   D7   -> LOX1 XSHUT  (left sensor shutdown, active LOW at boot)
//   A3   -> LOX2 XSHUT  (right sensor shutdown, active LOW at boot)
//   D2   <- IR receiver signal  (open-collector, INPUT_PULLUP, active LOW)
//   D13  -> Heartbeat LED  (built-in LED, toggles every 500 ms)
//   SDA  <-> VL53L0X 1 & 2 SDA
//   SCL  <-> VL53L0X 1 & 2 SCL
//   5V   -> VL53L0X VIN × 2, IR emitter VCC
//   GND  -> common ground
//
// -----------------------------------------------------------------------------
// I²C addresses
// -----------------------------------------------------------------------------
//   Both VL53L0X boards default to 0x29. At boot the Arduino holds LOX2
//   in reset (XSHUT low) while it reassigns LOX1 to 0x30, then brings
//   LOX2 out of reset and assigns it 0x31.
//
//     LOX1 (left  edge of arc)  ->  0x30
//     LOX2 (right edge of arc)  ->  0x31
//
// -----------------------------------------------------------------------------
// Serial output format  (one line per measurement, newline-terminated)
// -----------------------------------------------------------------------------
//     <left_mm> <right_mm> <beam_state>
//
//   left_mm, right_mm : int, distance in mm.  -1 if sensor reports RangeStatus=4
//                       (signal fail / out-of-range).
//   beam_state        : 0 = beam intact, 1 = beam broken (ball at center).
//
//   Example line:  142 163 0
//
// -----------------------------------------------------------------------------
// Library
// -----------------------------------------------------------------------------
//   Adafruit_VL53L0X   (install via Arduino Library Manager)
//
// -----------------------------------------------------------------------------
// Build / upload
// -----------------------------------------------------------------------------
//   arduino-cli compile --fqbn arduino:avr:uno dual_ir
//   arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyUSB0 dual_ir
// =============================================================================

#include "Adafruit_VL53L0X.h"

// ---- I²C addresses assigned at boot ----------------------------------------
#define LOX1_ADDRESS    0x30
#define LOX2_ADDRESS    0x31

// ---- Arduino pins ----------------------------------------------------------
#define SHT_LOX1        7       // LOX1 XSHUT
#define SHT_LOX2        A3      // LOX2 XSHUT
#define BEAM_PIN        2       // IR receiver signal (open-collector, active LOW)
#define HEARTBEAT_PIN   13      // Built-in LED

// ---- Timing ----------------------------------------------------------------
#define HEARTBEAT_INTERVAL_MS   500UL

Adafruit_VL53L0X lox1 = Adafruit_VL53L0X();
Adafruit_VL53L0X lox2 = Adafruit_VL53L0X();

bool          ledState  = false;
unsigned long lastBeat  = 0;

// -----------------------------------------------------------------------------
// heartbeat() - toggles the built-in LED so we can eyeball that the sketch is
// still running. Do not move into loop() body; it depends on millis() rolling
// over cleanly (which it does, because we use unsigned arithmetic).
// -----------------------------------------------------------------------------
void heartbeat() {
  if (millis() - lastBeat > HEARTBEAT_INTERVAL_MS) {
    ledState = !ledState;
    digitalWrite(HEARTBEAT_PIN, ledState);
    lastBeat = millis();
  }
}

// -----------------------------------------------------------------------------
// setID() - assign unique I²C addresses to both VL53L0X sensors.
//
// Both boards power up at the same default address (0x29). To use them on one
// bus we:
//   1. hold both XSHUT low (sensors in reset / high-Z on I²C),
//   2. release both XSHUT,
//   3. put LOX2 back in reset, initialise LOX1 at LOX1_ADDRESS,
//   4. release LOX2, initialise it at LOX2_ADDRESS.
//
// If either .begin() fails we loop forever - the host-side reader will see a
// timeout, log "Failed", and trigger a serial-port reset (see
// balancer/hardware/serial_reader.py::reset_sensor).
// -----------------------------------------------------------------------------
void setID() {
  digitalWrite(SHT_LOX1, LOW);
  digitalWrite(SHT_LOX2, LOW);
  delay(10);
  digitalWrite(SHT_LOX1, HIGH);
  digitalWrite(SHT_LOX2, HIGH);
  delay(10);

  // Bring LOX1 up first, keep LOX2 in reset.
  digitalWrite(SHT_LOX1, HIGH);
  digitalWrite(SHT_LOX2, LOW);
  if (!lox1.begin(LOX1_ADDRESS)) while (1);
  delay(10);

  // Now bring LOX2 out of reset and give it its own address.
  digitalWrite(SHT_LOX2, HIGH);
  delay(10);
  if (!lox2.begin(LOX2_ADDRESS)) while (1);

  // Timing budget is left at the library default (~33 ms). In single-shot
  // mode a smaller budget gave noisy readings; a larger one reduces rate too
  // much. Change deliberately if you also change SYSTEM.LEFT_CENTER_MM /
  // RIGHT_CENTER_MM in balancer/hardware/constants.py.
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(1);

  pinMode(SHT_LOX1,      OUTPUT);
  pinMode(SHT_LOX2,      OUTPUT);
  pinMode(BEAM_PIN,      INPUT_PULLUP);
  pinMode(HEARTBEAT_PIN, OUTPUT);

  setID();
}

void loop() {
  heartbeat();

  // IR beam: open-collector output, pulled up on the Arduino side.
  //   beam intact  -> HIGH -> we report 0
  //   beam broken  -> LOW  -> we report 1 (ball at center)
  int beam_state = !digitalRead(BEAM_PIN);

  VL53L0X_RangingMeasurementData_t measure1;
  VL53L0X_RangingMeasurementData_t measure2;

  // Single-shot ranging. See header for why we don't use continuous mode.
  lox1.rangingTest(&measure1, false);
  lox2.rangingTest(&measure2, false);

  Serial.print((measure1.RangeStatus != 4) ? measure1.RangeMilliMeter : -1);
  Serial.print(' ');
  Serial.print((measure2.RangeStatus != 4) ? measure2.RangeMilliMeter : -1);
  Serial.print(' ');
  Serial.println(beam_state);
}

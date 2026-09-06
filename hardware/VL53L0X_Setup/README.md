# VL53L0X Sensor Firmware (Arduino Uno)

Arduino-Uno firmware for the two VL53L0X time-of-flight distance sensors and
the Adafruit #2167 IR beam-break sensor mounted on the arc.

- **Deployed sketch:** [dual_ir/dual_ir.ino](dual_ir/dual_ir.ino) (single-shot ranging plus IR beam).

(Archived bring-up / diagnostic sketches have been removed from the
repository.)

The host-side parser for the deployed sketch lives in
**balancer/hardware/serial_reader.py** (function **create_distance_reader**).
Do not change the serial output format without updating that reader.

---

## Hardware

| Component | Qty | Notes |
|---|---|---|
| Arduino Uno | 1 | USB-attached. Under WSL2 (with `usbipd` forwarding the Uno from the Windows host) it shows up as **/dev/ttyACM0** (or `/dev/ttyUSB*` via an FTDI adapter). Install the udev rule in `balancer/hardware/` to get the stable alias **/dev/distance_sensor**. |
| AZ-Delivery VL53L0X breakout | 2 | One per arc edge; share the I²C bus. |
| Adafruit #2167 thru-beam IR (508 mm) | 1 pair | Emitter + receiver at arc apex. |

## Pin map

| Signal | Arduino pin | Connects to |
|---|---|---|
| LOX1 XSHUT | **D7** | VL53L0X #1 XSHUT |
| LOX2 XSHUT | **A3** | VL53L0X #2 XSHUT |
| IR beam signal | **D2** (INPUT_PULLUP, active LOW) | IR receiver open-collector output |
| Heartbeat LED | **D13** | Built-in LED, 0.5 s blink |
| I²C | **SDA**, **SCL** | Shared by both VL53L0X |
| Power | **5V**, **GND** | Sensors + IR emitter |

Full wiring table is in [BOM_AND_ASSEMBLY.md](../BOM_AND_ASSEMBLY.md#sensor-assembly).

## I²C addresses

Both VL53L0X boards power up at the default **0x29**. On boot the sketch holds
LOX2 in reset via its XSHUT pin, assigns LOX1 to **0x30**, then releases LOX2
and assigns it **0x31**.

| Sensor | Address | Position |
|---|---|---|
| LOX1 | **0x30** | Left edge of arc |
| LOX2 | **0x31** | Right edge of arc |

---

## Serial output: deployed sketch (115200 baud)

One line per measurement, space-separated, newline-terminated. Example line:

    142 163 0

Fields:

- **left_mm**: int, left (LOX1) sensor distance in mm. **-1** if out of range (RangeStatus == 4).
- **right_mm**: int, right (LOX2) sensor distance in mm. **-1** if out of range.
- **beam_state**: int, **0** = beam intact, **1** = beam broken (ball at center).

The host-side reader converts (left_mm, right_mm) to a ball angle in radians
using **LEFT_CENTER_MM**, **RIGHT_CENTER_MM**, and **ARC_RADIUS_M** from
**balancer/hardware/constants.py** (single source of truth; do not duplicate).

---

## Build and upload

The Arduino CLI is the expected tool, run from inside WSL2 (the Uno is
forwarded from the Windows host via `usbipd`). Install the Adafruit VL53L0X
library once:

    arduino-cli lib install "Adafruit VL53L0X"

Compile and upload the deployed sketch:

    cd hardware/VL53L0X_Setup
    arduino-cli compile --fqbn arduino:avr:uno dual_ir
    arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyACM0 dual_ir

(Replace `/dev/ttyACM0` with your board's device path; `ls /dev/ttyACM*
/dev/ttyUSB*` lists candidates after `usbipd attach`.)

Quick sanity check from the terminal:

    arduino-cli monitor -p /dev/ttyACM0 --config 115200

---

## Sketches in this folder

| Path | Status | Mode | IR beam | Purpose |
|---|---|---|---|---|
| **dual_ir/** | **deployed** | Single-shot | Yes | Production sketch: used by the host-side reader thread. |

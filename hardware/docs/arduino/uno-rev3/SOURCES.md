# Arduino Uno Rev 3 - documentation sources

**Part:** Arduino Uno Rev 3 - ATmega328P microcontroller board

**Role in this project:** Sensor-acquisition board. Reads two VL53L0X (I2C) and the Adafruit #2167 IR beam (D2), streams `left_mm right_mm beam_state` lines to the host at 115200 baud.

**Where it shows up in our code:** hardware/VL53L0X_Setup/dual_ir/dual_ir.ino; host-side parser in balancer/hardware/serial_reader.py (create_distance_reader)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| schematic.pdf | UNO-TH_Rev3e.sch (board schematic) | Rev3e | - | 1 |

## Upstream sources

- Vendor page: https://store.arduino.cc/products/arduino-uno-rev3
- Downloaded: 2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Pinout for D2 (INPUT_PULLUP), D7, A3, SDA/SCL, 5V/GND
- I2C bus behaviour / pull-ups on the Uno
- USB CDC enumeration (shows up as /dev/ttyUSB* with udev alias /dev/distance_sensor)

## Known gotchas

- Both VL53L0X boards boot at I2C address 0x29. The sketch holds LOX2 in reset (XSHUT low) while reassigning LOX1 to 0x30, then brings LOX2 up and assigns 0x31. If one XSHUT wire is loose at boot, both sensors collide on 0x29 - swap the XSHUT wire or run the `i2c_scanner` archived sketch to diagnose.
- The Uno's hardware UART is the same one used for the USB bridge, so any debug prints go over the same 115200 stream the host parser reads. Don't add Serial.println debug lines without also updating `create_distance_reader`.

# VL53L0X (ST) on AZ-Delivery breakout - documentation sources

**Part:** VL53L0X (ST) on AZ-Delivery breakout - Time-of-Flight distance sensor

**Role in this project:** Two units, one per arc edge, reporting ball distance over I2C (addresses 0x30 and 0x31 after XSHUT-based reassignment from the default 0x29). Single-shot ranging mode in the deployed sketch.

**Where it shows up in our code:**
- hardware/VL53L0X_Setup/dual_ir/dual_ir.ino (sensor acquisition)
- balancer/hardware/serial_reader.py (`create_distance_reader` - parses `left_mm right_mm beam_state`)
- balancer/hardware/sensor_utils.py (mm -> ball angle using `LEFT_CENTER_MM`, `RIGHT_CENTER_MM`, `ARC_RADIUS_M`)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | VL53L0X Datasheet - Time-of-Flight ranging sensor (ST, DS11555) | Rev 6 | - | 38 |
| breakout-datasheet.pdf | AZ-Delivery VL53L0X ToF breakout - Datenblatt (German) | - | - | 7 |
| an4846-using-multiple-vl53l0x-in-a-single-design-stmicroelectronics.pdf | AN4846 - Using multiple VL53L0X in a single design (ST, DocID029133) | Rev 1 | 2016-05 | 7 |
| an4907-cover-window-guidelines.pdf | AN4907 - VL53L0X ranging module cover window guidelines (ST, DocID029711) | Rev 3 | 2018-11 | 21 |

## Upstream sources

- AZ-Delivery product page: https://www.az-delivery.de/products/vl53l0x-time-of-flight-tof-laser-abstandssensor
- ST VL53L0X product page: https://www.st.com/en/imaging-and-photonics-solutions/vl53l0x.html
- AN4846:                    https://www.st.com/resource/en/application_note/an4846-using-multiple-vl53l0x-in-a-single-design-stmicroelectronics.pdf
- AN4907:                    https://www.st.com/resource/en/application_note/an4907-vl53l0x-ranging-module-cover-window-guidelines-stmicroelectronics.pdf
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Timing-budget vs. accuracy trade-off (datasheet §6 "Ranging modes")
- Range-status codes, especially **RangeStatus == 4** (reported as `-1` in our serial output)
- **AN4846 §2.1 "VL53L0X control management"** - the XSHUT-reset-then-reassign-I2C-address boot pattern that `dual_ir.ino::setID()` implements to map the two sensors to 0x30 / 0x31. (The Adafruit library's `begin(i2c_addr)` performs the `VL53L0X_SetDeviceAddress` call prescribed in §2.2 for us, so no manual API plumbing is needed.)
- Cross-interference behaviour when two sensors face the same workspace (drives our choice of **single-shot** ranging - see the `dual_ir_continuous` archived sketch for why continuous mode was dropped). Not covered by AN4846, which only addresses I2C-bus multiplexing.
- Cover-window / aperture guidance from AN4907 (relevant to the 3D-printed arc mount)

## Known gotchas

- The AZ-Delivery breakout ties XSHUT to a pull-up, so it comes up at 0x29 by default. The `dual_ir` sketch relies on this by holding the second sensor's XSHUT low at boot.
- Continuous ranging on both sensors simultaneously causes mutual interference (phantom ~200 mm readings); single-shot with sequential triggering avoids this. AN4846 does **not** discuss this - its focus is solely on sharing one I2C bus among multiple devices.
- AN4846 also sketches GPIO-expander variants (§2.1 Figure 3) for designs with many sensors on few host GPIOs. Not relevant to us (we have two dedicated Arduino pins, D7 and A3, as XSHUT lines).

# Adafruit #2167 IR beam-break sensor - documentation sources

**Part:** Adafruit #2167 - IR thru-beam sensor pair (508 mm range, open-collector receiver)

**Role in this project:** Single pair mounted at the apex of the arc. The emitter and receiver face each other across the ball path; beam-broken = ball-at-center. Used to zero the angle estimate and as a ground-truth center marker.

**Where it shows up in our code:**
- hardware/VL53L0X_Setup/dual_ir/dual_ir.ino (D2 with INPUT_PULLUP, active LOW - beam broken reports `1` on the serial line)
- balancer/hardware/serial_reader.py (`create_distance_reader` third field)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| learn-guide.pdf | Adafruit Learn - IR Breakbeam Sensors (tutorial + wiring) | - | 2024-06-03 | 8 |

## Upstream sources

- Product page: https://www.adafruit.com/product/2167
- Learn guide:  https://learn.adafruit.com/ir-breakbeam-sensors
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Receiver output type (**open-collector**, needs a pull-up on the Arduino side)
- Supply voltage range (5 V for both emitter and receiver)
- Beam alignment tips (sensitivity drops off fast beyond axis; the mount on the arc must be square)

## Known gotchas

- Without `INPUT_PULLUP` on the Arduino D2 pin, the receiver output floats and you get random triggers - the firmware sets pull-up at setup.
- Ambient IR (sunlight, some fluorescents, IR remotes) can false-trigger the receiver. Our setup is indoor + controlled lighting, so not an issue, but don't point a sunlit window at the arc.
- The "508 mm" rating is maximum range; at the arc apex distance (< 100 mm across the ball path) the beam is well inside its reliable zone.

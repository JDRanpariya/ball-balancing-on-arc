# SMC LEFB32NXS-1000 - documentation sources

**Part:** SMC LEFB32NXS-1000 - Electric actuator, belt-driven slider type, 1000 mm stroke, AC servo motor mount

**Role in this project:** The linear rail the cart rides on. The AM8121 drives the belt; the cart sits on the carriage. Arc radius, cart center, and stroke limits are what define `CART_LIMIT` in `balancer/hardware/constants.py`.

**Where it shows up in our code:** Mechanical; not referenced from code. Calibration (cart limits, stroke centers) lives in `balancer/hardware/constants.py` (`SystemConstants`).

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| catalogue.pdf | SMC LEF series electric actuator catalogue (LEFS ball-screw + LEFB belt) | - | 2013-04 | 143 |
| maintenance-manual.pdf | LEF Operation manual (LEFS/LEFB maintenance) | - | 2016-06-27 | 31 |

## Upstream sources

- SMC product page: https://www.smcworld.com/products/pickup/en-jp/electric_actuator/
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Dynamic load / max speed / max acceleration ratings (2000 mm/s max per the catalogue; we stay well below this)
- Belt-tension / re-tension procedure (maintenance manual)
- Mechanical drawings and mounting-hole layout for the base and the carriage
- Stroke limits and soft-stop locations
- **Equivalent lead, 54 mm/rev** (Series Variations table). The AC-servo variants (sizes 25, 32, 40) are 54 mm; the step-motor and 24 VDC-servo variants are 48 mm. Ours is the AC-servo mount, so 54 mm. This is the number that explains the axis-unit factor: see the length-unit note in `hardware/BOM_AND_ASSEMBLY.md`.
- Coupling dimensions for the motor-side shaft (for mating to the AM8121)

## Known gotchas

- The catalogue is from 2013 and covers several LEF variants - always confirm specs against the LEFB32 column specifically, not LEFS.
- Belt drift: over long runs, the belt can stretch slightly, shifting the "zero" carriage position relative to the absolute encoder. If the ball-center target drifts left/right after weeks of use, re-tension and re-home before editing `SystemConstants`.
- Soft-stops are mechanical bumpers, not electrical - the PLC's soft-limits (`-CART_LIMIT`, `+CART_LIMIT`) must be set tighter than the mechanical endpoints or the cart will slam the bumpers and fault the EL7201.

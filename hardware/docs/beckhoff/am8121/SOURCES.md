# AM8121-0FH0-0000 - documentation sources

**Part:** AM8121-0FH0-0000 - Servo motor, OCT, 4 A rated

**Role in this project:** Drives the LEFB32NXS belt-driven slide via a coupling. Controlled as AxisB by the EL7201 over a single OCT cable (power + feedback).

**Where it shows up in our code:** hardware/twincat/ball_on_arc/ (controlled as AxisB via EL7201)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | AM8121-wFyz - Servomotor 0.50 Nm (M0), F2 (58 mm) (product page) | - | - | 3 |
| operating-instructions.pdf | Operating instructions - AM8100 synchronous servomotors for compact drive technology | 2.4 | 2022-03-21 | 50 |

> `operating-instructions.pdf` is the AM8100 family manual; it covers AM8121-0FH0 as one of the variants. The product-page `datasheet.pdf` gives the SKU-specific torque/speed numbers.

## Upstream sources

- Vendor page: https://www.beckhoff.com/am8121
- SKU page:    https://www.beckhoff.com/am8121
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Nominal torque / speed / peak torque
- OCT cable pinout (motor phases + absolute-encoder feedback on the single orange cable)
- Coupling / shaft dimensions for mating to the LEFB32NXS
- Thermal / duty-cycle limits (relevant for long balancing runs)

## Known gotchas

- The `-0FH0-0000` suffix encodes winding, feedback, brake, and shaft options. Any swap to a different suffix invalidates the torque/speed numbers in `datasheet.pdf`; re-download the SKU page for the new variant.
- OCT cable is not hot-pluggable. Always power down the EL7201 before connecting/disconnecting.

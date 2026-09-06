# EL7201 - documentation sources

**Part:** EL7201 - 1-channel servo motor terminal, 48 V DC, 2.8 A rms rated (5.7 A rms peak for 1 s), OCT feedback

> The plain `EL7201` datasheet SKU actually ships with a **resolver**
> feedback interface. OCT is provided by the `EL7201-0010` variant (the
> "-0010" suffix encodes the OCT option). Our rig uses OCT, so the
> physical part is almost certainly `EL7201-0010` or another -xxxx OCT
> variant documented in the same `documentation.pdf`. **Confirm against
> the label on the terminal before ordering a spare.**

**Role in this project:** Drives the AM8121 over a single OCT cable. Exposed to the PLC as AxisB through TC2_MC2's `MC_MoveVelocity`. The brake chopper sits next to it on the stack (EL9576) and absorbs regen energy during hard decelerations.

**Where it shows up in our code:** hardware/twincat/ball_on_arc/ (NC axis AxisB; `FB_SerialCom_Vel` drives it via `MC_Power` / `MC_MoveVelocity` / `MC_Reset`)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | EL7201 - 1-ch servomotor, 48 V DC, 2.8 A, resolver (product page) | - | - | 3 |
| documentation.pdf | Documentation - Servo Motor Terminals EL7201-000x, EL7211-000x (48 V DC) | 3.8.0 | 2024-10-01 | 203 |

> The product page still lists "resolver" feedback; in our build the AM8121 uses **OCT** (One Cable Technology) instead. OCT is supported by the EL7201-0010 variant covered in the same `documentation.pdf`.

## Upstream sources

- Vendor page: https://www.beckhoff.com/el7201
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- OCT cable pinout and auto-detection of the AM8121
- CoE parameters that have to be tuned for the LEFB32NXS belt-drive inertia (current loop gains, velocity filter, max torque limit)
- Error register bit meanings (used by `FB_SerialCom_Vel.iState = 999 -> 0 -> 10 -> 20 -> 30` re-enable sequence)
- Brake control outputs (currently unused: the AM8121 variant on this rig has no holding brake)
- Integration with the EL9576 brake chopper on the same E-bus

## Known gotchas

- Motion parameters (`Acceleration = 15 000 mm/s^2`, `Deceleration = 15 000`, `Jerk = 50 000`) are set inside `FB_SerialCom_Vel`, not in the NC axis configuration. Changing them in one place without the other will cause `MC_MoveVelocity` to clamp or stall.
- The NC axis was commissioned with `ScaleFactorNumerator = 104.8576` against `ScaleFactorDenominator = 1048576`, i.e. 1e-4 axis units per encoder increment. At the terminal's default 20 single-turn bits (2^20 increments/rev) that is a feed constant of 104.8576 mm/rev, which is just the encoder count rescaled and was never the belt's real 54 mm/rev lead. Every reported position therefore runs ~1.94x physical. Do not "fix" this: all limits, datasets, identified dynamics, and published results are in these axis units consistently, so changing the scale factor invalidates them all. See the length-unit note in `hardware/BOM_AND_ASSEMBLY.md`.
- Always power down 24 V before connecting/disconnecting the OCT cable - hot-plug can corrupt the absolute-position counter.
- On a fresh TwinCAT project the EL7201 must be linked to an **NC axis** (AxisB here); linking to a CNC axis or a plain PLC axis will leave `MC_Power.Status = FALSE` permanently.

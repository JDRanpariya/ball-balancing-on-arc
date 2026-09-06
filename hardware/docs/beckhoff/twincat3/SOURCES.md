# TwinCAT 3 libraries - documentation sources

**Part:** TwinCAT 3 libraries - TC2_MC2 (motion control) and TF6340 (serial communication)

**Role in this project:** The two PLC libraries that `hardware/twincat/ball_on_arc/` depends on. TC2_MC2 provides `MC_Power`, `MC_Reset`, `MC_MoveVelocity` (used in `FB_SerialCom_Vel`). TF6340 provides `SerialLineControl`, `SendString`, `ReceiveString` (used in `BackgroundEL` + `FB_SerialCom_Vel`).

**Where it shows up in our code:** hardware/twincat/ball_on_arc/PLC_SerialCom_Sample1/PLC_SerialCom_Sample1.plcProj (library references section)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| tc2-mc2-library-reference.pdf | Manual - TwinCAT 3 PLC Library: Tc2_MC2 (TE1000) | 2.17.0 | 2026-03-23 | 167 |
| tf6340-serial-communication.pdf | Manual - TwinCAT 3 Serial Communication (TF6340) | 1.8.1 | 2026-03-25 | 146 |

> `TC2_SerialCom` function blocks used in the project (`SendString` / `ReceiveString`) are documented as part of the TF6340 manual; there is no separate TC2_SerialCom PDF.

## Upstream sources

- InfoSys index: https://infosys.beckhoff.com/
- TC2_MC2:  https://infosys.beckhoff.com/english.php?content=../content/1033/tcplclib_tc2_mc2/
- TF6340:   https://infosys.beckhoff.com/english.php?content=../content/1033/tf6340_tc3_serial_communication/
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- TC2_MC2: `MC_Power`, `MC_Reset`, `MC_MoveVelocity` (BufferMode = `MC_Aborting`), direction constants (`MC_Positive_Direction`, `MC_Negative_Direction`)
- TF6340: `SerialLineControl` (22-byte mode `SERIALLINEMODE_EL6_22B`), `SendString`, `ReceiveString`, buffer structure `ComBuffer`
- SerialCom InfoSys sample (this project is forked from it - keeps the `PLC_SerialCom_Sample1` naming)

## Known gotchas

- TC2_MC2 versions are pinned inside the PLC project. If you bump the library through XAE's library manager, re-verify that `MC_MoveVelocity` still accepts `BufferMode = MC_Aborting` on a `Busy` instance - earlier versions required `Execute` to fall to FALSE first.
- The original Beckhoff sample was written for **KL6001 / PcCOM**. We swapped in EL6002 X2 + 22-byte process image. Anywhere the POU comments still say KL6001 / EL6001 is historical and can be read as "EL6002 channel 2".

# Hardware Documentation Index

One folder per physical component. Each folder holds the vendor PDFs plus a
**SOURCES.md** that records the upstream URL, document version, which parts
of *our* code depend on the component, and any known gotchas.

If you're looking for how the components are assembled / wired together, see
[../BOM_AND_ASSEMBLY.md](../BOM_AND_ASSEMBLY.md).
If you're looking for the PLC project, see
[../twincat/README.md](../twincat/README.md). For Arduino firmware, see
[../VL53L0X_Setup/README.md](../VL53L0X_Setup/README.md).

---

## Control and drive (Beckhoff)

| Folder | Part | Role | Start here |
|---|---|---|---|
| [beckhoff/c6015/](beckhoff/c6015/)     | C6015-0010 | IPC running TwinCAT 3 | operating manual |
| [beckhoff/ek1100/](beckhoff/ek1100/)   | EK1100 | EtherCAT coupler (head of terminal stack) | coupler documentation |
| [beckhoff/ps2001/](beckhoff/ps2001/)   | PS2001-2410-0000 | 24 V DC PSU | datasheet |
| [beckhoff/zb8110/](beckhoff/zb8110/)   | ZB8110 | External 100 W / 10 Ω braking resistor for EL9576 | datasheet |
| [beckhoff/el7201/](beckhoff/el7201/)   | EL7201 | Servo drive (OCT, 4 A) | operating manual |
| [beckhoff/el9576/](beckhoff/el9576/)   | EL9576 | Brake chopper | datasheet |
| [beckhoff/el6002/](beckhoff/el6002/)   | EL6002 | 2-channel RS-232 terminal (host on X2 / channel 2) | documentation |
| [beckhoff/el1859/](beckhoff/el1859/)   | EL1859 | 8 DI + 8 DO combo | documentation |
| [beckhoff/am8121/](beckhoff/am8121/)   | AM8121-0FH0-0000 | Servo motor (OCT) | datasheet |
| [beckhoff/twincat3/](beckhoff/twincat3/) | TC2_MC2, TC2_SerialCom, TF6340 | TwinCAT libraries used by the PLC project | library references |

## Actuator

| Folder | Part | Role |
|---|---|---|
| [smc/lefb32nxs-1000/](smc/lefb32nxs-1000/) | LEFB32NXS-1000 | 1 m-stroke belt-driven linear slide |

## Sensors

| Folder | Part | Role |
|---|---|---|
| [sensors/vl53l0x/](sensors/vl53l0x/) | VL53L0X on AZ-Delivery breakout (×2) | Ball-position ToF sensors on the arc edges |
| [sensors/ir-beam-break-adafruit-2167/](sensors/ir-beam-break-adafruit-2167/) | Adafruit #2167 | IR thru-beam at the arc apex (ball-at-center) |

## Microcontroller

| Folder | Part | Role |
|---|---|---|
| [arduino/uno-rev3/](arduino/uno-rev3/) | Arduino Uno Rev 3 | Sensor acquisition board streaming to host over USB |

---

## Conventions

- **File names:** lowercase with hyphens (e.g. `operating-manual.pdf`). No
  spaces. No vendor prefix in the filename (the folder already carries it).
- **Each folder has a `SOURCES.md`.** Read this first. It ties the PDFs to
  our code.
- **PDFs up to ~20 MB** may be committed directly. Larger files go through
  Git-LFS (already set up in `.gitattributes`).
- **Never edit vendor PDFs.** If you need to annotate, create a sibling
  `notes.md`.
- **When swapping a component** (e.g. replacing the EL6001 with an EL6002),
  update the folder name, the BOM, the `SOURCES.md`, and - if the change
  affects code - the relevant TwinCAT POU / Arduino sketch / Python module.

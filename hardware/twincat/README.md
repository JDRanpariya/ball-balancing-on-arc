# TwinCAT 3: Ball on Arc Cart Control

TwinCAT 3 PLC project that drives the linear cart motor (AM8121 via EL7201)
and bridges velocity setpoints + axis telemetry to the host PC over an
EL6002 RS-232 terminal.

- **Active project:** [ball_on_arc/](ball_on_arc/) (velocity-mode bridge)
  (`FB_SerialCom_Vel` driving `MC_MoveVelocity`). Handles both discrete and
  continuous host actions through one protocol.

(The earlier `MC_Jog`-based variant has been removed from the repository.)

The host-side serial reader lives in
**balancer/hardware/serial_reader.py** (`create_ipc_reader`). All track
geometry and calibration constants come from
**balancer/hardware/constants.py** (`SystemConstants`). Do not duplicate
these values inside scripts or docs.

---

## System overview

```
[Host PC (Python)]
      | USB
      v
[USB <-> RS-232 adapter]
      | RS-232 (X2, channel 2)
      v
[EL6002 2-port RS-232 terminal] --- (EtherCAT E-bus) ---
      ^
      |
[EK1100 coupler] ---> [EL7201 servo drive] ---> [AM8121 motor] ---> [LEFB32NXS slide + cart]
      ^
      |
[C6015 IPC running TwinCAT 3]
```

- **EL6002 channel 2 (X2)** is the one wired to the PC. Channel 1 (X1) is
  unused. The process-image format used in code is the 22-byte variant
  (`EL6inData22B` / `EL6outData22B`, `SERIALLINEMODE_EL6_22B`), which is
  identical for EL6001 and EL6002, so legacy code/comments that name the
  EL6001 still work.
- `BackgroundEL` runs `SerialLineControl` in the fast task (PlcTask_Fast)
  to keep up with 115200 baud.
- `FB_SerialCom_Vel` runs in the standard task (PlcTask_Standard) and
  shares `TxBufferEL` / `RxBufferEL` with `BackgroundEL`.

---

## Hardware requirements

| Item | Role |
|---|---|
| Beckhoff IPC (C6015 or CX-series) running TwinCAT 3 | PLC runtime |
| EK1100 coupler | Head of EtherCAT stack |
| EL7201 + AM8121 (OCT) | Servo drive + motor |
| EL9576 brake chopper | Dissipates regen energy |
| **EL6002** (2-port RS-232) | Host communication; host wired to **channel 2 (X2)** |
| USB-to-RS-232 adapter | Host-side bridge to EL6002 X2 |
| SMC LEFB32NXS-1000 | Linear slide; axis configured as `AxisB` |

Full BOM + wiring lives in [BOM_AND_ASSEMBLY.md](../BOM_AND_ASSEMBLY.md).

---

## Serial protocol

**Baud / framing:** 115200, 8 data bits, no parity, 1 stop bit.
All payloads are wrapped with STX (0x02) / ETX (0x03) and an ASCII space
before and after the payload, e.g. `STX + " 450.123,12.500 " + ETX`.

### PLC -> Host  (every 10 ms)

    STX ' <ActPos_mm>,<ActVelo_mm_per_s> ' ETX

- Sent from `FB_SerialCom_Vel` via `SendString` at a 10 ms cadence.
- Host-side parser: `create_ipc_reader` in `balancer/hardware/serial_reader.py`.

### Host -> PLC  (on demand)

    STX ' <velocity_mm_per_s> ' ETX

- Single float; sign encodes direction (see `FB_SerialCom_Vel`):
  - **positive** value -> `MC_Negative_Direction`
  - **negative** value -> `MC_Positive_Direction`
  - **0** -> decelerate to standstill
- Discrete actions (e.g. left / stop / right) are sent as three fixed
  velocities (`-900`, `0`, `+900` mm/s in the current tuning). One block,
  one protocol, both modes.

---

## Motion parameters (set inside `FB_SerialCom_Vel`)

| Parameter | Value |
|---|---|
| Acceleration | 15 000 mm/s^2 |
| Deceleration | 15 000 mm/s^2 |
| Jerk         | 50 000 mm/s^3 |
| Telemetry TX period | 10 ms |
| BufferMode on each move | `MC_Aborting` (new setpoint immediately overrides the previous one) |

Two `MC_MoveVelocity` instances are alternated so a new setpoint can abort
the active one without waiting for its `Done` edge.

### Travel limits: `constants.py` is authoritative, not `axis_parameters.xml`

`axis_parameters.xml` is a snapshot of the NC axis taken during commissioning,
and its `SoftEndMinControl` / `SoftEndMaxControl` ranges (200 and 1750 axis
units) are slightly tighter than the limits the rig actually ran with. The
benchmark trials reach 197.5 and 1759.2, about 9 axis units (~5 mm of physical
travel) past the exported soft end. The limits that governed every published
run are `TRACK_START_M = 0.200` and `TRACK_END_M = 1.753` in
`balancer/hardware/constants.py`, i.e. `CART_LIMIT = 0.7765` about the centre.
Take those as the operating envelope; treat the XML's soft ends as indicative.

---

## Using the project

### 1. Open the solution

1. Double-click **ball_on_arc/SerialCom_InfoSys_Sample1.sln**; TwinCAT XAE
   (Visual Studio shell) opens with all POUs, GVLs, and references loaded.
2. POUs present in the active project:
   - `MAIN` (instantiates `FB_SerialCom_Vel`)
   - `Background`/`BackgroundEL` (drive the EL6002 via `SerialLineControl`)
   - `FB_SerialCom_Vel` (the velocity bridge)

### 2. Pick the target PLC

1. In the XAE toolbar, open the target selector and choose
   **Choose Target System -> Search (Ethernet)**.
2. Enter the PLC's AMS Net ID or IP, broadcast-search, pick it, click OK.
3. TwinCAT router on the PLC must be reachable (UDP/TCP 48898).

### 3. Configure the EL6002 terminal in System Manager

- Baud rate: 115200, 8 data bits, no parity, 1 stop bit.
- Process-image mode: **22-byte** (matches `SERIALLINEMODE_EL6_22B` in
  `BackgroundEL`).
- Link the PDOs of **channel 2** to `stIn_EL6001` (inputs) and
  `stOut_EL6001` (outputs). Channel 1 can be left unlinked.
- Map `Background` (which calls `BackgroundEL`) to **PlcTask_Fast**
  (<=1 ms cycle) for reliable throughput.

### 4. Activate -> Login -> Start

| Step | Action |
|---|---|
| Activate Configuration | toolbar: hammer + gear icon (or F8). Accepts the RunMode restart prompt. |
| Login | PLC menu -> Login (or green plug icon). |
| Start | PLC menu -> Start (or green play icon). |
| Stop  | PLC menu -> Stop. |
| Logout | PLC menu -> Logout. |

The program only executes when both **Login** and **Start** have been run.

---

## Host-side

The host-side code is not inside this folder any more. Use:

- **balancer.hardware.robot.Robot** (Gym-compatible hardware interface)
- **balancer.hardware.serial_reader** (background reader threads)
- **balancer.hardware.action_utils** (serial-command encoders)

Serial port and track geometry defaults live in
`balancer/hardware/serial_config.py` and `balancer/hardware/constants.py`.
Override the serial ports via the `BALANCER_IPC_PORT` and
`BALANCER_DIST_PORT` environment variables.

---

## Troubleshooting

**No telemetry from PLC**

- Confirm TwinCAT is in **Run** mode (green status indicator).
- Confirm the EL6002 baud rate matches 115200 and channel 2 PDOs are linked.
- Confirm the host USB-to-RS-232 adapter is on the expected port
  (default `/dev/ttyUSB0`; override with `BALANCER_IPC_PORT`).

**Cart does not move**

- Check `McPower.Status = TRUE` in the PLC online view.
- If the axis is in error, `FB_SerialCom_Vel` cycles through
  `iState := 999 -> 0 -> 10 -> 20 -> 30` to re-enable it; watch `iState`.
- Confirm no hardware limit switch is triggered.

**TwinCAT cannot find the PLC**

- Host and PLC on the same subnet.
- TwinCAT router running on the PLC.
- Windows Firewall allows TCP/UDP 48898.

**Host-side buffer overflow messages**

- The Python reader's buffer grew past 1024 bytes. This usually means the
  PLC is transmitting but Python isn't draining the port fast enough
  (an unhandled exception in the reader thread, or the thread is blocked
  elsewhere).

# Hardware

The physical rig: bill of materials, assembly guide, vendor documentation,
and the firmware / PLC code that runs on the embedded targets.

This folder covers the **physical and embedded side** of the project.
Host-side Python (Gym environment, readers, controllers) lives under
[../balancer/](../balancer/) and imports all hardware-dependent constants
from there. Calibration values and serial config are not duplicated inside
this folder.

---

## Top-level contents

| Path | What's in it |
|---|---|
| [BOM_AND_ASSEMBLY.md](BOM_AND_ASSEMBLY.md) | Bill of materials, wiring diagrams, DIN-rail layout, power-on sequence. |
| [docs/](docs/) | Vendor documentation (datasheets, manuals): one folder per physical part. |
| [twincat/](twincat/) | TwinCAT 3 PLC project. The active project is [twincat/ball_on_arc/](twincat/ball_on_arc/). See [twincat/README.md](twincat/README.md). |
| [VL53L0X_Setup/](VL53L0X_Setup/) | Arduino firmware for the ball-position ToF sensors + IR beam break. Active sketch is [VL53L0X_Setup/dual_ir/](VL53L0X_Setup/dual_ir/). See [VL53L0X_Setup/README.md](VL53L0X_Setup/README.md). |
| [assets/](assets/) | 3D-printed cart STL files + reference photos. |

---

## Where calibration and protocol constants live

Reference these from the code that needs them; do not hard-code values in
`hardware/`.

| What | Source of truth |
|---|---|
| Arc radius, cart limits, sensor centers | `balancer/hardware/constants.py` (`SystemConstants`) |
| Dynamics params (masses, friction, etc.) | `balancer/core/params.py` (`DEFAULT_PARAMS`) |
| Serial ports, baud, timeouts, STX/ETX framing | `balancer/hardware/serial_config.py` (`SerialConfig`) |

---

## Sign conventions (cart direction)

The linear motor's encoder polarity is **inverted** relative to the rig's
left/right labelling: a *positive* TwinCAT axis velocity (`MC_Positive_Direction`)
moves the cart **left**, a *negative* one moves it **right**. Three layers of
sign handling align the physical motion with the Python/model convention
where **positive = right** everywhere.

| Layer | Where | Convention |
|---|---|---|
| **PLC** (`FB_SerialCom_Vel.TcPOU`) | `parsedVelocity >= 0` -> `MC_Negative_Direction` -> cart **right**; `< 0` -> `MC_Positive_Direction` -> cart **left** | positive command = right |
| **Action encoding** (`balancer/hardware/action_utils.py`) | `encode_continuous_action(action, max_value=900)` -> `0.9 * 900 = +810` sent as `STX +810.000 ETX` | positive action = right |
| **Serial reader** (`balancer/hardware/serial_reader.py`) | `position = track_center - pos_m` (flips encoder), `velocity = -vel_mps` (negates encoder) | positive pos/vel = right |
| **Dataset / model / PPO** | state = `[cart_pos, cart_vel, ball_pos, ball_vel]`, action ∈ [-1, 1] | positive = right throughout |

**Full chain for one command (action = +0.9):**

1. PPO outputs `+0.9`.
2. `Robot.step()` -> `get_action_command(0.9, "cont")` -> `encode_continuous_action(0.9, 900)` -> `b'\x02 810.000 \x03'`.
3. PLC parses `+810` -> `MC_Negative_Direction` -> cart moves **right**.
4. Encoder reports decreasing `ActPos` -> `pos_m` falls; `ActVelo` is negative -> `vel_mps < 0`.
5. Serial reader stores `position = track_center - pos_m` (increases) and `velocity = -vel_mps` (positive) -> state shows cart moving **right**, consistent with the command.

> **Velocity-negation history.** The `velocity = -vel_mps` negation was
> introduced after the first data-collection campaign. Datasets collected
> **before** that change (notably the `arcball_cont_1M_pre_neg.h5` set,
> collected on a separate host with the non-negated reader) store
> raw encoder velocity, where **positive = left**. Models trained on the
> pre-negation dataset therefore embed the opposite cart-velocity sign
> from models trained on the post-negation dataset. See
> `paper/ram/supplementary/sections/G_world_model.tex` for which checkpoint
> each PPO/MPPI result uses.

---

## Typical workflows

### Connect the rig to the PC (WSL2 USB passthrough)

The rig connects to the host PC via two USB serial devices: the Beckhoff
EL6002 channel (RS-232 over a USB-FTDI adapter) for the motor controller,
and the Arduino Uno (USB, CH340 on clones / CDC-ACM on originals) for the
ToF/IR sensor board. WSL2 does not expose host USB devices by default;
use the [usbipd-win](https://github.com/dorssel/usbipd-win) project to
forward them.

1. **On the Windows host.** Install usbipd-win (latest release), open
   PowerShell as Administrator, and bind each device:

   ```powershell
   usbipd list                    # find the BUSIDs
   usbipd bind --busid <BUSID>    # e.g. 1-1 for the FTDI, 1-2 for the Arduino
   usbipd wsl list                # confirm both show "Attached" after step 3
   ```

2. **Custom WSL2 kernel (one-time).** The stock WSL2 kernel lacks
   USB_ACM, USB_SERIAL, and the FTDI / CH341 serial drivers. Rebuild the
   kernel from https://github.com/microsoft/WSL2-Linux-Kernel with these
   flags set in `.config`:

   ```text
   CONFIG_USB=y
   CONFIG_USB_ACM=y              # CDC-ACM (Arduino Uno -> /dev/ttyACM*)
   CONFIG_USB_SERIAL=y
   CONFIG_USB_SERIAL_FTDI_SIO=y  # FTDI-based USB-serial adapters
   CONFIG_USB_SERIAL_CH341=y     # CH340/CH341 (common on Arduino clones)
   CONFIG_USBIP_CORE=m
   CONFIG_USBIP_VHCI_HCD=m
   CONFIG_USBIP_HOST=m
   CONFIG_USBIP_VUDC=m
   ```

   Build and install the kernel, then point `%USERPROFILE%\.wslconfig`
   at it (`kernel=<path>`) and restart WSL (`wsl --shutdown`).

3. **Attach in WSL2.** From PowerShell, forward each device:

   ```powershell
   usbipd wsl attach --busid <BUSID>
   ```

   Inside WSL2 the devices appear as `/dev/ttyUSB*` (FTDI/CH340) or
   `/dev/ttyACM*` (Arduino Uno); verify with `ls /dev/tty*`. The
   Python stack expects the sensor board at the stable alias
   `/dev/distance_sensor` (the default in
   [`balancer/balancer/hardware/serial_config.py`](../balancer/balancer/hardware/serial_config.py));
   create it with a udev rule, e.g.:

   ```text
   # /etc/udev/rules.d/99-distance-sensor.rules
   SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="distance_sensor"
   ```

   (`1a86:7523` is the CH340; use `lsusb` to confirm your IDs, then
   `sudo udevadm control --reload && sudo udevadm trigger`.)

**Troubleshooting.** If a device shows "Detached" after `usbipd bind`,
run `usbipd wsl attach --busid <BUSID>` again after opening a WSL2
shell; attach binds to a running WSL instance. Both devices must be
attached concurrently during hardware operation.

### Flash the Arduino sensor board

1. Open [VL53L0X_Setup/dual_ir/dual_ir.ino](VL53L0X_Setup/dual_ir/dual_ir.ino).
2. `arduino-cli compile --fqbn arduino:avr:uno VL53L0X_Setup/dual_ir`
3. `arduino-cli upload --fqbn arduino:avr:uno -p /dev/ttyUSB1 VL53L0X_Setup/dual_ir`
4. Confirm live output with `arduino-cli monitor -p /dev/ttyUSB1 --config 115200`.

### Deploy a new PLC build

1. Open [twincat/ball_on_arc/SerialCom_InfoSys_Sample1.sln](twincat/ball_on_arc/SerialCom_InfoSys_Sample1.sln) in TwinCAT XAE.
2. Target selector -> pick the PLC over Ethernet (see [twincat/README.md](twincat/README.md)).
3. Activate Configuration (F8) -> Login -> Start.

### Document a new component

1. Add a folder under [docs/](docs/) (vendor-folder/part-folder).
2. Add vendor PDFs inside.
3. Fill in `SOURCES.md`.
4. Add a BOM row in [BOM_AND_ASSEMBLY.md](BOM_AND_ASSEMBLY.md).

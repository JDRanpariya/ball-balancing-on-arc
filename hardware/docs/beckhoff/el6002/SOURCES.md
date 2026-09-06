# EL6002 - documentation sources

**Part:** EL6002 - 2-channel RS-232 terminal (X1, X2)

**Role in this project:** Host-PC serial bridge. Host is wired to **channel 2 (X2)**; channel 1 (X1) is unused. Uses the 22-byte process image (`EL6inData22B` / `EL6outData22B`, `SERIALLINEMODE_EL6_22B`). Baud 115200, 8N1.

**Where it shows up in our code:**
- hardware/twincat/ball_on_arc/PLC_SerialCom_Sample1/POUs/BackgroundEL.TcPOU (runs `SerialLineControl` on PlcTask_Fast)
- hardware/twincat/ball_on_arc/PLC_SerialCom_Sample1/POUs/FB_SerialCom_Vel.TcPOU (uses `SendString` / `ReceiveString` against the shared buffers)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | EL6002 - 2-ch RS-232 communication interface, D-sub (product page) | - | - | 3 |
| documentation.pdf | Documentation - Serial Interface Terminals EL600x, EL602x | 6.1.0 | 2026-03-25 | 238 |

> The same `documentation.pdf` also covers the EL6001 (1-ch). Comments in older TwinCAT POUs still name the EL6001 because the 22-byte process image is identical; that is expected and not a bug.

## Upstream sources

- Vendor page: https://www.beckhoff.com/en-en/products/i-o/ethercat-terminals/el6xxx-communication/el6002.html
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Process-image modes (**22-byte** vs 1024-byte); we use 22-byte
- PDO linking per channel (only channel 2 PDOs mapped to `stIn_EL6001` / `stOut_EL6001`)
- Baud / parity / stop-bit configuration via CoE
- D-sub X1 / X2 pinout (TX, RX, GND, handshake lines)

## Known gotchas

- Only **X2 / channel 2** is wired. Channel 1 (X1) PDOs can be left unlinked; if they are linked but no device is connected, `SerialLineControl` will still work but will log framing errors on channel 1.
- `BackgroundEL` must run in the fast task (<=1 ms cycle). If it is mapped to `PlcTask_Standard` (10 ms), the host-side Python reader will periodically see buffer-overflow warnings at 115200 baud.
- When switching the process-image mode in System Manager, the PDO layout changes and `SERIALLINEMODE_EL6_22B` in `BackgroundEL` must match - otherwise `FB_SerialCom_Vel.iState` will spin on the enable step.

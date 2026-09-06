# EK1100 - documentation sources

**Part:** EK1100 - EtherCAT coupler (head of the terminal stack)

**Role in this project:** Converts the 100BASE-TX uplink from the C6015 into E-bus signals for the EL-series terminals. No process data of its own.

**Where it shows up in our code:** Not referenced directly from Python or PLC code - configured inside the TwinCAT System Manager as the head of the I/O tree.

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | EK1100 - EtherCAT Coupler (product page) | - | - | 3 |
| documentation.pdf | Documentation - EK110x-00xx, EK15xx EtherCAT Bus Coupler | 4.8.0 | 2025-12-01 | 104 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/ek1100
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Power supply wiring (24 V DC into the coupler; separate terminal-bus vs power-contact supplies)
- RJ-45 IN / OUT port orientation (IN = from C6015, OUT = daisy chain to next coupler if any)
- Maximum number of EtherCAT terminals per coupler (we use < 10, well under the 65 535 limit)

## Known gotchas

- EK1100 has no battery / no diagnostic process data of its own. If the RUN LED is off, check 24 V on the power-contact input, not the E-bus.
- Swapping the position of the EK1100 inside the stack changes the EtherCAT address of every downstream terminal; re-scan the I/O tree in TwinCAT after any physical reshuffle.

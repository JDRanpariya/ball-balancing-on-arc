# PS2001-2410-0000 - documentation sources

**Part:** PS2001-2410-0000 - Power supply PS2000; 24 V DC / 10 A / 240 W output, 1-phase AC 100-240 V input

**Role in this project:** Primary 24 V DC supply for the EtherCAT stack (EK1100 power contacts, EL6002, EL1859, EL7201 logic, EL9576 logic) and anything downstream on the 24 V rail.

**Where it shows up in our code:** Passive component - not referenced from PLC or Python code.

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | PS2001-2410-0000 - Power supply PS2000, 24 V DC 10 A (product page) | - | - | 2 |
| documentation.pdf | Documentation - Power supply 24 V DC, 10 A, 1-phase, AC 100-240 | 1.2.0 | 2024-08-22 | 45 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/ps2001-2410-0000
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Output current rating (10 A) - comfortably above the stack's worst-case draw
- Extra-power / inrush behaviour for tripping fuses in short-circuit conditions
- DC OK relay contact wiring (currently unused but available for future watchdog)
- Approvals relevant for our lab environment (ATEX, DNV, SEMI F47)

## Known gotchas

- Wide-range input (AC 100-240 V **or** DC 110-150 V). On a DC input, polarity matters; double-check before energizing.
- The DC OK LED is the fastest way to tell whether the 24 V output is actually up; a dim PS2001 status LED can also mean the PFC is holding in soft-start (happens when you cycle power too quickly).

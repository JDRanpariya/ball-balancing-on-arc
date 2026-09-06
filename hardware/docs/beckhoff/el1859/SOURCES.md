# EL1859 - documentation sources

**Part:** EL1859 - 8-channel digital input + 8-channel digital output (24 V DC, 3 ms, 0.5 A)

**Role in this project:** Spare digital I/O terminal on the stack. Reserved for limit switches, status LEDs, and auxiliary signals. Not currently wired to any load - present on the DIN rail for future expansion.

**Where it shows up in our code:** Not referenced from the TwinCAT project or any Python module at this time.

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | EL1859 - 8-ch DI + 8-ch DO, 24 V DC, 3 ms, 0.5 A (product page) | - | - | 3 |
| documentation.pdf | Documentation - Digital HD Input/Output Terminals EL18xx | 2.9.0 | 2025-12-04 | 158 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/en-en/products/i-o/ethercat-terminals/el1xxx-digital-input/el1859.html
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Channel pinout on the 16-pin connector (inputs vs. outputs, common supply)
- Process-image layout for linking to TwinCAT variables if/when we wire it up
- 0.5 A per-channel / 4 A total current limit (important if driving relays/solenoids)

## Known gotchas

- EL1859 combines DI and DO in a single terminal; the channel numbering is not continuous (inputs 1-8 vs. outputs 1-8 use different PDO sub-indices). Re-link PDOs after any firmware update.
- Outputs are push-pull, not open-collector. Don't parallel them with another source driving the same rail.

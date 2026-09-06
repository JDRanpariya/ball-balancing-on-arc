# EL9576 - documentation sources

**Part:** EL9576 - Brake chopper terminal (with external ZB8110 braking resistor)

**Role in this project:** Dissipates regenerative energy from the EL7201 during hard decelerations to keep the DC link voltage under its over-voltage threshold. Sits next to the EL7201 on the DIN rail.

**Where it shows up in our code:** Not referenced from the TwinCAT PLC code - configured in the System Manager as an I/O terminal next to the EL7201.

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | EL9576 - Brake chopper EtherCAT Terminal (product page) | - | - | 3 |
| documentation.pdf | Documentation - Brake Chopper Terminal with EtherCAT connection EL9576 | 2.8.0 | 2024-11-20 | 122 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/en-en/products/i-o/ethercat-terminals/el9xxx-system/el9576.html
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Chopper switch-on / switch-off voltage thresholds (configured via CoE)
- Max continuous vs. peak regen power ratings (we rely on the ZB8110's 100 W continuous limit)
- Wiring between EL9576 and the external ZB8110 resistor

## Known gotchas

- The EL9576 alone has no large internal resistor - without the **ZB8110** connected, extended regen events will trip it on thermal warning.
- The DC link it protects is the **EL7201 load-voltage rail** (8…48 V DC external supply feeding the EL7201's motor phases, per the EL7201 datasheet). It does not protect the 24 V logic supply feeding the EK1100 / EL6002 / EL7201 logic; that rail has no dedicated protection beyond the PS2001's own output circuitry.
- The ZB8110 wired to this terminal is the **external braking resistor** (10 Ω / 100 W), not a surge or buffer module. Earlier drafts of `hardware/BOM_AND_ASSEMBLY.md` mislabelled ZB8110 as a 24 V surge protector; that was wrong and has been corrected - see `docs/beckhoff/zb8110/SOURCES.md`.

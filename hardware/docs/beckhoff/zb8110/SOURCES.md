# ZB8110 - documentation sources

**Part:** ZB8110 - External braking resistor, 100 W continuous, 10 Ohm

**Role in this project:** External dissipation resistor for the EL9576 brake chopper. Burns off regen energy pumped back into the DC link during rapid cart decelerations. Connected directly to the EL9576's resistor terminals.

**Where it shows up in our code:** Passive component - not referenced from PLC or Python code.

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | ZB8110 - External braking resistor (product page) | - | - | 2 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/en-en/search-results/?q=ZB8110
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Rated power (100 W continuous) and energy-pulse ratings (4 kJ / 1.2 s at 1% duty, 8 kJ / 7.2 s at 6% duty)
- Resistance value (10 Ohm) - must match what the EL9576's chopper expects
- Mounting plate / clearance requirements (resistor gets hot)
- Terminal wiring to the EL9576

## Known gotchas

- This is **not a DIN-rail terminal** and does not clip onto the EtherCAT stack. It has its own mounting plate and is wired to the EL9576's resistor terminals via a 1 m lead. Earlier revisions of `hardware/BOM_AND_ASSEMBLY.md` misdescribed it as a 24 V surge-protection module on the PSU side; that was wrong and has since been corrected.
- There is **no separate 24 V surge-protection device** on this rig - the 24 V rail is protected only by the PS2001's own output circuitry.
- Gets hot in use (casing rated up to 250 °C during sustained braking) - keep flammable material away and do not cover the mounting plate.

# C6015-0010 - documentation sources

**Part:** C6015-0010 - Ultra-compact IPC, Intel Atom (4 cores)

**Role in this project:** Runs the TwinCAT 3 runtime and the EtherCAT master.

**Where it shows up in our code:** hardware/twincat/ball_on_arc/ (entire TwinCAT project runs here)

---

## Files in this folder

| File | Document / title | Version | Date | Pages |
|---|---|---|---|---|
| datasheet.pdf | C6015-0010 - Ultra-compact industrial PC (product page) | - | - | 4 |
| manual.pdf | Manual - C6015 Industrial PC | 4.5 | 2026-04-23 | 47 |

## Upstream sources

- Vendor page: https://www.beckhoff.com/c6015
- Downloaded:  2026-04-27
- Downloaded by: Jaydeepkumar Ranpariya

## Key sections we rely on

- Mechanical drawing / mounting orientation (for DIN-rail + back-panel mounts)
- Technical data / power draw (24 V input sizing)
- BIOS / initial bring-up
- EtherCAT master port assignment (which X001/X002 port the EK1100 uplink goes into)

## Known gotchas

- The C6015 has a fan-less passive-cooled design; don't block the side vents when installing in a narrow control cabinet - the CPU will thermal-throttle TwinCAT cycles.
- The built-in EtherCAT master is on the X001 port. Using X002 puts you on the general Ethernet NIC and EtherCAT will not enumerate.

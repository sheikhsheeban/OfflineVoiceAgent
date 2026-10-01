# Kin clip firmware (HW1, HW2, HW3, HW7)

**Status: written, not compiled or flashed.** The laptop software talks to the clip through the serial
protocol below and is tested against a simulated clip (`tests/test_w4_ids.py`). The hardware rows of the
week-4 table (HW1, HW2, HW3, E01, E02, E03, HW7 on real LEDs) can only be checked on a bench with the
board, using the checklist at the end of this file.

## Parts

| Part | Role | Notes |
|---|---|---|
| ESP32-S3 dev board | USB device: microphone + serial | native USB pins GPIO19/20 |
| INMP441 I²S MEMS mic | HW1 | SCK GPIO4, WS GPIO5, SD GPIO6, L/R to GND |
| SPDT slide switch + P-MOSFET (e.g. AO3401) | HW2 | the switch gates the MOSFET that powers the mic's 3V3 pin |
| Green LED + 330 Ω | HW3 | wired to the **same switched 3V3 rail** as the mic, not to a GPIO: lit only when the mic has power |
| WS2812 RGB LED | HW7 | data on GPIO48 |
| Mute sense | E05 | GPIO7 reads the switch (pull-up; low = muted) so the firmware can report it |

Why the LED is on the mic's rail: firmware can lie, wiring cannot. If the green LED is on, the mic has power.

## Serial protocol (USB CDC, 115200, ASCII lines)

| Direction | Line | Meaning |
|---|---|---|
| clip → host | `HELLO kin-clip 1` | sent when the port opens |
| clip → host | `M1` / `M0` | mute switch on / off (sent on change and every 2 s) |
| host → clip | `Lgreen_solid` | listening |
| host → clip | `Lgreen_pulse` | capturing a command |
| host → clip | `Lblue_pulse` | thinking |
| host → clip | `Lamber_blink` | fetching online |
| host → clip | `Lred_solid` | error |
| host → clip | `Loff` | muted (the WS2812 goes dark; the green mic LED is dark by hardware) |

The USB product string is `Kin Clip`; the laptop finds the microphone and the serial port by that name
(`kin/audio/clip_serial.py`, `kin/audio/capture.py`).

## Build

```
cd firmware
idf.py set-target esp32s3
idf.py build flash monitor
```

Needs ESP-IDF 5.x with the `esp_tinyusb` component (UAC2 microphone + CDC).

## Bench checklist (the parts software cannot prove)

1. Plug in: the laptop lists a microphone named "Kin Clip" (`uv run kin devices`) and `kin run` says `listening on Kin Clip`.
2. Slide the switch to mute: the green LED goes dark **with the board's firmware halted** (hold RESET). This proves HW3 is hardware.
3. With the switch on, a multimeter on the mic's 3V3 pin reads 0 V (HW2).
4. With the switch on, the UI shows "Muted by the clip's switch"; clicking Unmute in the UI does nothing (F4).
5. Switch off: audio resumes without restarting Kin.
6. Unplug the clip while Kin runs: Kin switches to the laptop microphone and says so.

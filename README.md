# Orchard row guidance (Intel RealSense D455)

Prototype for farm-tractor guidance when GNSS is degraded under orchard canopy. It uses the regular geometry of tree rows — parallel lines, a vanishing point at the end of the row, and known in-row spacing — with an Intel RealSense D455.

Two outputs:

1. **Left / right steering cue** — stay in the alley and aimed at the gap between the two trees at the end of the row.
2. **Velocity** — track trunks in metric depth, and use planted tree spacing as a second speed estimate.

Python is for bring-up (simulator + web lightbar). C++ is the Raspberry Pi 5 runtime.

## Will this be easy?

A working prototype is straightforward. A field-ready GNSS backup is not.

The D455 is a reasonable sensor (global shutter, ~87° FOV, useful depth about 0.6–6 m, sometimes ~8–10 m outdoors). It will **not** depth-see the end of a 100 m row. Far-field aiming uses **RGB vanishing perspective**; near-field left/right offset uses **depth to the nearest trunks**. Treat this as an aid while GNSS is noisy, not a centimeter-level replacement.

Sign convention: **positive lateral error means the tractor is right of row center → steer left.**

## Python simulator (no camera)

```bash
pip install -r requirements.txt
python app.py --source sim
```

Open http://127.0.0.1:5000

```bash
python -m pytest -q
```

## RealSense D455

On the machine with the camera (USB 3):

```bash
pip install -r requirements.txt pyrealsense2
python app.py --source realsense --row-width 4.5 --tree-spacing 3.5
python app.py --source realsense --bag capture.bag
```

Mount ~1.4–1.7 m high, looking slightly down the row (`camera_pitch_deg ≈ -6`).

## C++ on a Raspberry Pi 5

Stay at **848×480 @ 15 FPS**, skip `rs2::align`, use a 5 V / 5 A supply and an active cooler.

```bash
sudo apt install libopencv-dev cmake pkg-config build-essential
# Build librealsense from source for ARM, then:
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build cpp/build -j4
./cpp/build/test_guidance
./cpp/build/tractor_guidance --source sim --no-gui --frames 90
./cpp/build/tractor_guidance --source realsense --gui
```

## Clone on Windows (Hack 2026)

```bat
cd /d C:\Users\sullric\Documents\Hack 2026
git clone https://github.com/sullricTrimble/orchard.git
```

That creates `C:\Users\sullric\Documents\Hack 2026\orchard`.

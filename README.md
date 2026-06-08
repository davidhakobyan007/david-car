# Fast Object Tracker

Opens your webcam, lets you pick an object by clicking on it (or drawing a box
around it), and follows it with a square in real time.

Built with Python + OpenCV. Tested target: **Ubuntu 24.04**.

## Setup (Ubuntu 24.04)

```bash
# 1. System packages (camera + GUI support)
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 v4l-utils

# 2. Get the project into a virtual environment
cd david-car
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate    # if not already active
python3 tracker.py
```

A window opens showing your camera. **Click on an object** — a green square
locks on and follows it. For a tighter fit, **click and drag** a box instead.

## Controls

| Key / Action          | What it does                                  |
|-----------------------|-----------------------------------------------|
| Left click            | Place a tracking box on the click point       |
| Left click + drag     | Draw a custom tracking box                    |
| `+` / `-`             | Grow / shrink the default click box           |
| `r`                   | Reset — stop tracking, pick a new object      |
| `c`                   | Cycle tracker (KCF → MOSSE → CSRT)            |
| `m`                   | Toggle mirror (flipped) camera view           |
| `q` or `ESC`          | Quit                                          |

If the tracker loses the object, it **automatically searches the frame and
re-locks** on it (box turns orange while searching). The default tracker is
**CSRT**, the most robust. Switch to a faster one with `c` or `--tracker`.

## Speed vs. accuracy

The tracker algorithm controls how "fast" it feels:

| Tracker | Speed       | Accuracy | Notes                          |
|---------|-------------|----------|--------------------------------|
| MOSSE   | Very fast   | Lower    | Best when you want max speed   |
| KCF     | Fast        | Good     | Default — solid all-rounder    |
| CSRT    | Slower      | Highest  | Best for small/tricky objects  |

For the fastest possible tracking:

```bash
python3 tracker.py --tracker MOSSE
```

## Options

```bash
python3 tracker.py --camera 0 --tracker KCF --box-size 120
python3 tracker.py --width 640 --height 480   # lower resolution = higher FPS
```

- `--camera N` — pick a different webcam (try `1`, `2`, ... if `0` is wrong).
  Run `v4l2-ctl --list-devices` to see your cameras.
- `--tracker {KCF,MOSSE,CSRT}` — tracking algorithm.
- `--box-size N` — default square size (pixels) for a single click.
- `--width` / `--height` — capture resolution. Lower it for more FPS.

## Tips for faster tracking

- Use `--tracker MOSSE`.
- Lower the resolution: `--width 640 --height 480`.
- Good, even lighting helps the tracker hold the lock.
- If the box drifts off, press `r` and re-select.

## Troubleshooting

- **"Could not open camera index 0"** — another app is using the webcam, or
  the index is wrong. Close other apps; try `--camera 1`.
- **No window appears / Qt errors** — install GUI libs: `sudo apt install libgl1`.
- **Permission denied on /dev/video0** — add yourself to the video group:
  `sudo usermod -aG video $USER` then log out and back in.

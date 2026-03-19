# Virtual DJ 🎧✋

Virtual DJ controller you control with your hands. No hardware, just a webcam. The long-term idea is to project a DDJ-FLX4 layout onto a desk with a laser projector and use a camera to track your hands on it — but right now we're just getting the hand tracking solid first. Honestly this all stemmed from me just wanting a weekend project to vibe out some cool hand tracking stuff. Turns out.. its quite hard.

## How it works

Your hands are tracked to 2 cursors on the screen allowing you to interact with a virtual board. Pinch your fingers together to grab whatever the cursor is hovering over, then:

- **Rotate your wrist** → spins the jogwheel or turns the knob
- **Move your hand up/down** → slides a fader
- **Let go of the pinch** → drops the control

Controls on the board right now: 2 jogwheels (Deck A/B), filter and hi EQ knobs for each deck, volume faders, and a crossfader.

## Setup

```bash
conda create -n handdj python=3.11 -y
conda activate handdj
pip install -r requirements.txt
python hand_dj_cursor.py
```

If you have multiple cameras or need to flip it:

```bash
python hand_dj_cursor.py --camera 1
python hand_dj_cursor.py --flip-camera
```

## Requirements

- Python 3.9+
- A webcam

# Hand DJ Controller 🎧✋

Air-DJ prototype that uses your webcam to track hand gestures and map them to a virtual DJ interface with 2 jogwheels and 2 knobs.

## Requirements

- Python 3.9+
- A webcam
- macOS, Windows, or Linux

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python hand_dj_controller.py
```

## How It Works

The app uses **MediaPipe Hands** to detect 21 hand landmarks per hand in real-time from your webcam. It then interprets gestures and maps them to DJ controls:

### Gesture Controls

| Gesture | What It Does |
|---------|-------------|
| **Fist** (left hand) | Grabs Deck A jogwheel — rotate your wrist to spin |
| **Fist** (right hand) | Grabs Deck B jogwheel — rotate your wrist to spin |
| **Point** (left hand) | Grabs Filter A knob — move finger up/down to turn |
| **Point** (right hand) | Grabs Filter B knob — move finger up/down to turn |
| **Open hand** | Releases all controls |

### UI Layout

```
┌─────────────────────────────────────────┐
│              HAND DJ                     │
│          ┌──────────────┐               │
│          │  WEBCAM FEED │               │
│          └──────────────┘               │
│                                          │
│   ╭─────╮                  ╭─────╮      │
│   │ JOG │   DECK A    B   │ JOG │      │
│   │  A  │                  │  B  │      │
│   ╰─────╯                  ╰─────╯      │
│                                          │
│   (KNOB A)                (KNOB B)      │
│                                          │
│  L: FIST ●              ● R: POINT      │
└─────────────────────────────────────────┘
```

## Troubleshooting

- **"Cannot open webcam"**: Make sure no other app is using the camera. On macOS, grant Terminal camera permissions in System Preferences → Privacy.
- **Laggy tracking**: Ensure good lighting. MediaPipe works best with even, bright light and a plain background.
- **Wrong hand detection**: The app mirrors the camera, so your left hand controls Deck A (left side). If hands swap, try moving them further apart.

## Next Steps (Future Development)

- [ ] Add virtual MIDI output (python-rtmidi) to connect to Rekordbox/Mixxx
- [ ] Map jogwheel spin to actual pitch/scratch MIDI CC messages
- [ ] Add crossfader gesture (two-hand pinch)
- [ ] Add EQ knobs (3-band per deck)
- [ ] Add play/pause gesture (tap motion)
- [ ] Add volume faders (vertical hand slide)
- [ ] Calibration mode for hand size/distance

## Architecture

```
Webcam → MediaPipe Hands → Gesture Classifier → DJ Control Mapper → Pygame UI
                                                          ↓
                                               (Future: MIDI Output)
                                                          ↓
                                               (Rekordbox / Mixxx)
```

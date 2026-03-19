# Virtual DJ 🎧✋

A virtual DJ kit modeled after the **Pioneer DDJ-FLX4** controller. The end goal: a laser projector displays the DJ board layout onto a surface, and a camera tracks your hand movements on that projected board — letting you mix, scratch, and tweak knobs on a fully virtual controller with no physical hardware.

## The Vision

Instead of spending hundreds on a physical controller, you project one. A laser projector throws the DDJ-FLX4 layout (jogwheels, faders, knobs, pads) onto a desk or table. An overhead camera watches your hands interact with the projected controls — pinching a knob, spinning a jogwheel, sliding a fader — and translates those gestures into real MIDI signals that drive software like Rekordbox or Mixxx.

## Where We Are Now

We're in **Phase 1: hand tracking proof-of-concept**. Before wiring up a projector and dialing in spatial calibration, we need to make sure the core gesture recognition is solid. Right now the app runs with a standard webcam and a Pygame-rendered DJ interface on screen so we can test how well it detects and responds to hand movements over the virtual controls.

**Current prototype includes:**
- 2 jogwheels (Deck A / Deck B) controlled by making a fist and rotating
- 2 filter knobs controlled by pointing and moving your finger up/down
- Real-time hand landmark tracking via MediaPipe (21 points per hand)
- Live webcam feed with tracking overlay
- Left hand controls Deck A, right hand controls Deck B

## Roadmap

**Phase 1 — Webcam + Screen (current)**
- [x] Hand detection and gesture classification (fist, point, open)
- [x] Jogwheel grab and rotation tracking
- [x] Knob grab and value control
- [ ] Add remaining DDJ-FLX4 controls (crossfader, EQ, volume faders, play/pause/cue pads)
- [ ] Virtual MIDI output via python-rtmidi to connect to Rekordbox/Mixxx
- [ ] Calibration mode for hand size and distance

**Phase 2 — Projector Display**
- [ ] Replace on-screen UI with laser projector output mapped to a physical surface
- [ ] Spatial calibration system to align projected layout with camera tracking
- [ ] Adjust gesture detection for top-down camera angle

**Phase 3 — Full Integration**
- [ ] Complete DDJ-FLX4 MIDI mapping so the virtual board is a drop-in replacement
- [ ] Low-latency optimization for live performance use
- [ ] Portable setup (compact projector + camera rig)

## Requirements

- Python 3.9+
- A webcam
- macOS, Windows, or Linux

## Quick Start

```bash
# Create a conda environment
conda create -n handdj python=3.11 -y
conda activate handdj

# Install dependencies
pip install -r requirements.txt

# Run
python hand_dj_controller.py
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CURRENT (Phase 1)                                           │
│                                                              │
│  Webcam → MediaPipe Hands → Gesture Classifier → Pygame UI  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  FUTURE (Phase 2-3)                                          │
│                                                              │
│  Laser Projector → Surface (desk/table)                      │
│                        ↑ hands interact                      │
│  Overhead Camera → MediaPipe → Gesture Classifier            │
│                                       ↓                      │
│                                 MIDI Output                  │
│                                       ↓                      │
│                              Rekordbox / Mixxx               │
└──────────────────────────────────────────────────────────────┘
```

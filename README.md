# SupportRole — Realtime Predictive AI Assistant

An ultra-low-latency, fully-local realtime AI assistant for Windows.

It captures live system (loopback) audio, transcribes it incrementally with
Faster-Whisper on CUDA, streams the rolling transcript into a local LLM
served by Ollama, and renders streaming keyword/hint output on a transparent,
always-on-top overlay — **before** the speaker finishes talking.

Target hardware: **RTX 4080 SUPER + Ryzen 9 7950X3D + 64 GB RAM, Windows 11**.

---

## Latency Budget

| Stage                      | Target           |
| -------------------------- | ---------------- |
| Audio chunk → VAD          | < 20 ms          |
| Partial transcript visible | 300–700 ms       |
| First LLM token            | 700–1500 ms      |
| Full hint stream           | < 2 s end-to-end |

---

## Architecture

```
 ┌──────────────────┐    ┌──────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────┐
 │ Audio source     │──► │ VAD  │──► │ Faster-Whisper   │──► │ Rolling Ctx  │──► │ Ollama   │
 │  loopback / mic /│    │ WebRTC│   │ small.en / FP16  │    │ debouncer    │    │ stream   │
 │  UDP receiver    │    └──────┘    └──────────────────┘    └──────────────┘    └────┬─────┘
 └──────────────────┘                                                                  │
                                                                                       ▼
                                                                              ┌──────────────┐
                                                                              │ PyQt6 Overlay│
                                                                              │ transparent  │
                                                                              └──────────────┘
```

All stages run concurrently via `asyncio` + bounded queues. Every queue
drops stale items so the pipeline always works on the freshest audio.

### Input modes

Pick one in `AudioConfig.input_mode` ([support_role/config.py](support_role/config.py)):

| Mode       | Source                                                    | When to use                                            |
| ---------- | --------------------------------------------------------- | ------------------------------------------------------ |
| `loopback` | Default speaker captured via WASAPI loopback              | Transcribe whatever your PC is **playing** (Zoom, YouTube, Discord). Audio still plays normally. |
| `mic`      | Default microphone                                        | Transcribe **yourself** speaking.                       |
| `udp`      | int16 PCM stream over UDP (default `0.0.0.0:50007`)       | Receive audio **from another PC / device** over the network. Matches the simple `socket.sendto` sender script. |

List available devices:

```powershell
python -m support_role.tools.list_devices
```

UDP defaults match this sender (int16 stereo 48 kHz, 1024-frame packets):

```python
sock.sendto(int16_stereo_48k.tobytes(), ("<this-pc-ip>", 50007))
```

---

## Folder Structure

```
SupportRole/
├── README.md
├── requirements.txt
├── run.pyw                       # silent launcher (no console)
├── run.py                        # debug launcher (with console)
├── packaging/
│   └── build_exe.ps1             # PyInstaller build script
├── docs/
│   ├── INSTALL.md
│   ├── CUDA_SETUP.md
│   ├── OLLAMA_SETUP.md
│   └── PERFORMANCE.md
└── support_role/
    ├── __init__.py
    ├── config.py                 # central tunables (input_mode, UDP port, …)
    ├── main.py                   # async orchestrator + tray
    ├── pipeline/
    │   ├── __init__.py
    │   ├── audio_capture.py      # WASAPI loopback / mic (soundcard)
    │   ├── udp_receiver.py       # UDP int16 PCM receiver
    │   ├── vad.py                # WebRTC VAD streaming gate
    │   ├── transcriber.py        # Faster-Whisper streaming
    │   ├── context_buffer.py     # rolling transcript + debouncer
    │   ├── llm_streamer.py       # Ollama streaming client
    │   └── util_queue.py         # latest-wins async queue
    ├── tools/
    │   └── list_devices.py       # CLI: enumerate speakers / mics
    └── ui/
        ├── __init__.py
        ├── overlay.py            # PyQt6 transparent always-on-top overlay
        └── tray.py               # system tray icon
```

See [docs/INSTALL.md](docs/INSTALL.md) to get started.

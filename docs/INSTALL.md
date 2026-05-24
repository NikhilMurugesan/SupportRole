# Installation

## 1. Python

Install **Python 3.12 (64-bit)** for Windows. Make sure `python` and
`pythonw` are on `PATH`.

```powershell
python --version
```

## 2. Create venv

```powershell
cd C:\Users\mmmmn\SupportRole
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
```

## 3. Install CUDA prerequisites

See [CUDA_SETUP.md](CUDA_SETUP.md).

## 4. Install Python deps

```powershell
pip install -r requirements.txt
```

If `webrtcvad-wheels` fails to install, try:

```powershell
pip install webrtcvad-wheels --only-binary=:all:
```

## 5. Install & start Ollama

See [OLLAMA_SETUP.md](OLLAMA_SETUP.md). The default model is `phi3:mini`.

## 6. Choose your audio input

Edit `AudioConfig` in [../support_role/config.py](../support_role/config.py):

```python
input_mode: Literal["loopback", "mic", "udp"] = "udp"  # default
device_name: Optional[str] = None                       # substring match, or None
```

| Mode       | What it captures                                          |
| ---------- | --------------------------------------------------------- |
| `loopback` | Whatever your PC is **playing** through its default speaker (WASAPI loopback). Audio still plays normally. |
| `mic`      | Your **microphone** (default or matched by `device_name`). |
| `udp`      | int16 PCM packets received on `UdpConfig.listen_ip:listen_port` (default `0.0.0.0:50007`). Matches the simple `socket.sendto` sender script. |

List devices so you can copy a name fragment into `device_name`:

```powershell
python -m support_role.tools.list_devices
```

### UDP-specific setup

When `input_mode="udp"`:

1. **Allow the port through Windows Firewall** (only needed once):

   ```powershell
   New-NetFirewallRule -DisplayName "SupportRole UDP" `
       -Direction Inbound -Protocol UDP -LocalPort 50007 -Action Allow
   ```

   If you get `WinError 10013` on startup, the port is inside Windows'
   excluded dynamic range (typically reserved by Hyper-V/WSL/Docker).
   Check with:

   ```powershell
   netsh int ipv4 show excludedportrange protocol=udp
   ```

   Pick any port outside the listed ranges and set
   `UdpConfig.listen_port` accordingly.

   (Or just click **Allow** the first time Windows pops up the prompt.)

2. On the sending machine, point your sender script at this PC's IP and
   port `50007` (or whatever you set `UdpConfig.listen_port` to).
   The expected format is **int16 PCM, stereo, 48 kHz**
   (matching the reference sender). The receiver downmixes to mono and
   resamples to 16 kHz automatically.

3. Tune `UdpConfig` in [../support_role/config.py](../support_role/config.py)
   if your sender uses a different port, sample rate, or channel count.

## 7. Run

Debug (with console + logs):

```powershell
python run.py
```

You should see one of:

```
Capture mode=loopback source=Speakers (Realtek…)
Capture mode=mic source=Microphone (Realtek…)
UDP audio receiver listening on 0.0.0.0:50007 (src=48000Hz/2ch -> 16000Hz mono)
```

Silent (no console):

```powershell
pythonw run.pyw
```

The overlay appears in the bottom-right corner; a tray icon provides
**Quit**.

# Performance tuning guide

All knobs live in [support_role/config.py](../support_role/config.py).

## Latency contributors (typical, RTX 4080 SUPER + 7950X3D)

| Stage                          | Cost            |
| ------------------------------ | --------------- |
| Loopback capture chunk (200ms) | 200 ms (intrinsic) |
| WebRTC VAD                     | < 1 ms          |
| Faster-Whisper `small.en` FP16 | 20–40 ms / call |
| Context debounce               | 350 ms          |
| Ollama first token (`phi3:mini`) | 250–500 ms    |
| Qt signal -> repaint           | < 16 ms         |

End-to-end: **first hint visible ~1.0–1.5 s after speech onset**.

## Knobs that matter

### Capture chunk size — `AudioConfig.capture_chunk_ms`
* Smaller (100 ms): lower latency, more Python overhead.
* Larger (400 ms): slightly higher latency, smoother CPU.
* For `input_mode="udp"` this is the chunk size **after** receiving and
  resampling — the receiver buffers raw datagrams (≈21 ms each at the
  default 1024-frame block) until it has a full chunk. Drop to 100 ms
  if you want tighter end-to-end latency over the network.

### UDP jitter — `UdpConfig.socket_rcvbuf`
* Only relevant when `input_mode="udp"`.
* Defaults to 1 MiB — large enough to absorb several hundred ms of
  network jitter without dropping packets.
* Raise it if you see audio glitches on a flaky LAN/Wi-Fi link.

### VAD aggressiveness — `AudioConfig.vad_aggressiveness`
* `2` (default): good for clean podcasts/meetings.
* `3`: noisy environments. Drops more borderline frames.

### Whisper model — `WhisperConfig.model_size`
* `tiny.en`: ~2× faster than small, lower accuracy.
* `small.en` (default): best speed/accuracy on a 4080 SUPER.
* `medium.en`: 2–3× slower, only worth it if you raise `transcribe_interval_ms`.

### Transcribe cadence — `WhisperConfig.transcribe_interval_ms`
* Lower = more frequent partial updates, more GPU work.
* `250 ms` is a sweet spot for `small.en` on this GPU.

### Debounce — `LLMConfig.debounce_ms`
* Lower = LLM fires more often (more aborts, more visible "keyword churn").
* Higher = fewer hints but each one is more stable.

### Context window — `LLMConfig.context_chars`
* The whole point: keep prompts **short**. 200–400 chars keeps prompt
  evaluation under ~50 ms for `phi3:mini`.

### `num_predict` — `LLMConfig.num_predict`
* Caps output tokens. 40 is plenty for "≤15 words" output.

### Cancel-on-new-input — `LLMConfig.cancel_on_new_input`
* `True` (default): aborts in-flight generations when a fresher
  transcript arrives. This is the single biggest win for perceived
  latency — users never see hints derived from stale audio.

## Profiling

Enable debug logs:

```powershell
python run.py
```

Look for lines like:

```
LLM done seq=42 in 480ms (38 chars)
```

If first-token latency > 1 s consistently:

1. Verify Ollama is using GPU (`nvidia-smi` shows VRAM usage).
2. Switch to `phi3:mini` if you were on `llama3:8b`.
3. Lower `context_chars` to 200.

If transcript lags speech:

1. Lower `capture_chunk_ms` to 150.
2. Verify `compute_type="float16"` (not int8).
3. Confirm Whisper is on CUDA (log line at startup).

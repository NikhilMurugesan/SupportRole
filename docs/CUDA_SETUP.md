# CUDA setup for Faster-Whisper on RTX 4080 SUPER

Faster-Whisper uses **CTranslate2**, which ships with prebuilt CUDA
binaries. You do **not** need to install the full CUDA Toolkit, but you
**do** need matching CUDA runtime + cuDNN DLLs on `PATH`.

## Required components

| Component        | Version (CT2 ≥ 4.4)       |
| ---------------- | ------------------------- |
| NVIDIA driver    | ≥ 550 (Game Ready or Studio) |
| CUDA runtime     | 12.x                      |
| cuDNN            | 9.x for CUDA 12           |

## Recommended: use the prebuilt cuDNN wheels

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

These ship the required DLLs and CT2 will find them automatically when
imported from inside the venv.

## Verify

```powershell
python -c "from faster_whisper import WhisperModel; m=WhisperModel('small.en', device='cuda', compute_type='float16'); print('OK')"
```

If you see `CUDA failed with error ...`, your driver is too old — update
to the latest NVIDIA Studio Driver.

## FP16 settings

The defaults already use:

* `device="cuda"`
* `compute_type="float16"`
* `beam_size=1`
* `temperature=0.0`
* `condition_on_previous_text=False`

On the RTX 4080 SUPER, `small.en` transcribes ~1 s of audio in **20–40 ms**.

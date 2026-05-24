# Ollama setup

## Install

Download and install Ollama for Windows: <https://ollama.com/download>.

After install, Ollama runs a background service on
`http://127.0.0.1:11434`.

## Pull a model

Default (fast, ~3.8 GB):

```powershell
ollama pull phi3:mini
```

Alternative (better hints, ~5.7 GB, still fits comfortably on 16 GB VRAM):

```powershell
ollama pull llama3:8b-instruct-q4_K_M
```

To switch models, edit `LLMConfig.model` in
[support_role/config.py](../support_role/config.py).

## Verify GPU usage

```powershell
ollama run phi3:mini "say hi"
```

In another terminal:

```powershell
nvidia-smi
```

You should see the `ollama` process pinned to the GPU.

## Tuning for low first-token latency

Ollama keeps the model "hot" if it has been used recently. We send
`"keep_alive": "30m"` on every request to ensure the model stays
resident in VRAM between utterances.

`num_predict` is hard-capped at 40 tokens in
[support_role/config.py](../support_role/config.py) so generations
finish quickly even if the model tries to ramble.

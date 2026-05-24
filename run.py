"""Debug launcher (with console window for logs)."""

from support_role.main import run

if __name__ == "__main__":
    # debug=False keeps the per-chunk DEBUG spam (UDP rx stats, audio chunk
    # -> VAD, VAD got chunk) out of the console. Set to True only when you
    # need to investigate the audio path frame-by-frame.
    raise SystemExit(run(debug=False))

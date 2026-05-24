"""List available audio devices.

Usage:
    python -m support_role.tools.list_devices
"""

from __future__ import annotations

import soundcard as sc


def main() -> None:
    print("=== Speakers (use these for loopback) ===")
    default_spk = sc.default_speaker()
    for s in sc.all_speakers():
        marker = "*" if s.name == default_spk.name else " "
        print(f" {marker} {s.name}")

    print("\n=== Microphones ===")
    default_mic = sc.default_microphone()
    for m in sc.all_microphones(include_loopback=False):
        marker = "*" if m.name == default_mic.name else " "
        print(f" {marker} {m.name}")

    print("\n(* = system default)")
    print("\nCopy a substring of the name you want into AudioConfig.device_name")
    print("and set AudioConfig.input_mode to 'mic' or 'loopback'.")


if __name__ == "__main__":
    main()

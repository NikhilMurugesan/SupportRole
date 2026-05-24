"""UDP audio receiver.

Compatible with this sender:

    sock.sendto(int16_stereo_48k.tobytes(), (ip, 50007))

We:
  * receive raw int16 PCM datagrams on a background thread,
  * downmix stereo -> mono,
  * resample 48 kHz -> 16 kHz (integer 3x decimation with averaging),
  * convert to float32 in [-1, 1],
  * push fixed-size chunks (matching AudioConfig.capture_chunk_ms) into the
    same `LatestWinsQueue` that the loopback/mic capture uses, so the rest
    of the pipeline is unchanged.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Optional

import numpy as np

from ..config import CONFIG, AudioConfig, UdpConfig
from .util_queue import LatestWinsQueue

log = logging.getLogger(__name__)


class UdpAudioReceiver:
    def __init__(
        self,
        out_queue: LatestWinsQueue,
        audio_cfg: AudioConfig = CONFIG.audio,
        udp_cfg: UdpConfig = CONFIG.udp,
    ) -> None:
        self.audio_cfg = audio_cfg
        self.udp_cfg = udp_cfg
        self.out = out_queue
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sock: Optional[socket.socket] = None

        # Pre-compute resampling parameters.
        src_sr = udp_cfg.source_sample_rate
        dst_sr = audio_cfg.sample_rate
        if src_sr % dst_sr != 0:
            raise ValueError(
                f"UDP source rate {src_sr} must be an integer multiple of "
                f"pipeline rate {dst_sr}. Adjust UdpConfig or AudioConfig."
            )
        self._decim = src_sr // dst_sr  # e.g. 48000 / 16000 = 3

        # Target chunk size at destination sample rate.
        self._chunk_dst = int(dst_sr * audio_cfg.capture_chunk_ms / 1000)
        # Float32 mono accumulator at destination rate.
        self._accum = np.zeros(0, dtype=np.float32)
        self._chunks_pushed = 0

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._open_socket()
        self._thread = threading.Thread(
            target=self._run, name="UdpAudioReceiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---------------------------------------------------------------- internal
    def _open_socket(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.udp_cfg.socket_rcvbuf)
        except OSError:
            pass
        try:
            s.bind((self.udp_cfg.listen_ip, self.udp_cfg.listen_port))
        except PermissionError as exc:
            # Windows raises WinError 10013 either when (a) another process
            # is already bound to this UDP port, or (b) the port falls inside
            # the dynamic range reserved by Hyper-V/WSL/Docker.
            raise PermissionError(
                f"Cannot bind UDP {self.udp_cfg.listen_ip}:{self.udp_cfg.listen_port} "
                f"({exc}).\n"
                f"  1. Make sure no other process (e.g. your test receiver "
                f"script) is already listening on this port.\n"
                f"  2. If still failing, run:\n"
                f"       netsh int ipv4 show excludedportrange protocol=udp\n"
                f"     and pick a port outside the listed ranges, then set\n"
                f"     UdpConfig.listen_port in support_role/config.py."
            ) from exc
        except OSError as exc:
            raise OSError(
                f"Cannot bind UDP {self.udp_cfg.listen_ip}:{self.udp_cfg.listen_port} "
                f"({exc}). Another process may already be listening on this port."
            ) from exc
        s.settimeout(0.5)
        self._sock = s
        log.info(
            "UDP audio receiver listening on %s:%d (src=%dHz/%dch -> %dHz mono)",
            self.udp_cfg.listen_ip,
            self.udp_cfg.listen_port,
            self.udp_cfg.source_sample_rate,
            self.udp_cfg.source_channels,
            self.audio_cfg.sample_rate,
        )

    def _run(self) -> None:
        sock = self._sock
        assert sock is not None
        ch = self.udp_cfg.source_channels
        recv_buf = self.udp_cfg.recv_buffer

        import time
        first_packet = True
        pkt_count = 0
        byte_count = 0
        last_stats_t = time.monotonic()
        start_t = time.monotonic()
        warned_no_packets = False
        seen_addrs: set[tuple] = set()

        log.info("UDP receiver thread running. Waiting for first packet…")

        while not self._stop.is_set():
            try:
                packet, addr = sock.recvfrom(recv_buf)
            except socket.timeout:
                # No data this tick — surface a one-time warning after 5 s of silence.
                if not warned_no_packets and (time.monotonic() - start_t) > 5.0:
                    log.warning(
                        "No UDP packets received on %s:%d after 5 s. "
                        "Check that the sender is targeting THIS PC's IP and "
                        "port=%d, and that Windows Firewall allows inbound UDP.",
                        self.udp_cfg.listen_ip,
                        self.udp_cfg.listen_port,
                        self.udp_cfg.listen_port,
                    )
                    warned_no_packets = True
                continue
            except OSError:
                if self._stop.is_set():
                    return
                log.exception("UDP recv failed")
                return
            if not packet:
                continue

            pkt_count += 1
            byte_count += len(packet)
            if first_packet:
                log.info(
                    "FIRST UDP audio packet: %d bytes from %s", len(packet), addr
                )
                first_packet = False
                warned_no_packets = True  # don't fire the warning anymore
                seen_addrs.add(addr)
            elif addr not in seen_addrs:
                # Log new sources ONCE (the sender may use multiple ephemeral
                # source ports — that's normal, we just want to know they exist).
                log.info("Additional UDP source seen: %s", addr)
                seen_addrs.add(addr)

            try:
                self._handle_packet(packet, ch)
            except Exception:
                log.exception("Failed to process UDP audio packet")

            # Periodic stats — every ~1 s of wall time. Logged at DEBUG to
            # avoid spamming the console; flip the logger to DEBUG to see them.
            now = time.monotonic()
            if now - last_stats_t >= 1.0:
                kbps = (byte_count * 8) / 1000.0 / (now - last_stats_t)
                log.debug(
                    "UDP rx: %d pkt/s, %.1f kbps, sources=%d",
                    pkt_count, kbps, len(seen_addrs),
                )
                pkt_count = 0
                byte_count = 0
                last_stats_t = now

    def _handle_packet(self, packet: bytes, channels: int) -> None:
        # int16 PCM, interleaved by channel.
        audio_i16 = np.frombuffer(packet, dtype=np.int16)
        if audio_i16.size == 0:
            return
        # Drop trailing partial frame if the datagram was odd-sized.
        usable = (audio_i16.size // channels) * channels
        if usable == 0:
            return
        if channels > 1:
            frames = audio_i16[:usable].reshape(-1, channels)
            # Downmix to mono via mean. int32 to avoid overflow.
            mono_i16 = frames.astype(np.int32).mean(axis=1).astype(np.int32)
        else:
            mono_i16 = audio_i16[:usable].astype(np.int32)

        # Resample 48k -> 16k by averaging groups of `_decim` samples
        # (cheap low-pass + decimation).
        d = self._decim
        n = (mono_i16.size // d) * d
        if n == 0:
            return
        decimated = mono_i16[:n].reshape(-1, d).mean(axis=1)
        # Normalize to float32 [-1, 1].
        float_mono = (decimated.astype(np.float32) / 32768.0).astype(np.float32)

        # Append and flush in fixed-size chunks so VAD timing is stable.
        self._accum = np.concatenate([self._accum, float_mono])
        chunk = self._chunk_dst
        while self._accum.size >= chunk:
            out = np.ascontiguousarray(self._accum[:chunk], dtype=np.float32)
            self._accum = self._accum[chunk:]
            rms = float(np.sqrt(np.mean(out * out))) if out.size else 0.0
            self._chunks_pushed += 1
            # Only log the very first chunk at INFO so we know audio is
            # flowing; everything after that is DEBUG-only.
            if self._chunks_pushed == 1:
                log.info(
                    "audio chunk -> VAD: #%d, %d samples (%d ms), rms=%.4f",
                    self._chunks_pushed, out.size,
                    int(out.size * 1000 / self.audio_cfg.sample_rate), rms,
                )
            else:
                log.debug(
                    "audio chunk -> VAD: #%d, %d samples (%d ms), rms=%.4f",
                    self._chunks_pushed, out.size,
                    int(out.size * 1000 / self.audio_cfg.sample_rate), rms,
                )
            self.out.put_latest(out)

"""Runtime LLM source and Bedrock secret handling.

The static defaults live in config.py. User-selected runtime preferences are
stored in support_role/config.local.json, while the Bedrock API key is stored
in the project .env file, which is ignored by git.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import CONFIG

LLMSource = Literal["local", "online"]

LOCAL_SOURCE: LLMSource = "local"
ONLINE_SOURCE: LLMSource = "online"
BEDROCK_API_KEY_ENV = "AWS_BEARER_TOKEN_BEDROCK"


@dataclass(frozen=True)
class LLMSettingsSnapshot:
    source: LLMSource
    bedrock_region: str
    bedrock_model_id: str
    bedrock_api_key: str

    @property
    def active_model_label(self) -> str:
        if self.source == LOCAL_SOURCE:
            return CONFIG.llm.model
        return self.bedrock_model_id

    @property
    def active_source_label(self) -> str:
        if self.source == LOCAL_SOURCE:
            return "Local LLM"
        return "Bedrock LLM"


class LLMSettingsStore:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.env_path = root / CONFIG.online_llm.env_file
        self.runtime_config_path = root / CONFIG.online_llm.runtime_config_file
        self._lock = threading.RLock()

    def snapshot(self) -> LLMSettingsSnapshot:
        with self._lock:
            env_values = self._read_env()
            state = self._read_runtime_config()

            source = self._clean_source(
                str(state.get("source") or CONFIG.online_llm.default_source)
            )
            bedrock_api_key = self._secret(BEDROCK_API_KEY_ENV, env_values)

            # Boto3 automatically picks up AWS_BEARER_TOKEN_BEDROCK from the
            # process environment. Mirror .env into os.environ for this run.
            if bedrock_api_key:
                os.environ[BEDROCK_API_KEY_ENV] = bedrock_api_key

            return LLMSettingsSnapshot(
                source=source,
                bedrock_region=self._value(
                    "SUPPORTROLE_BEDROCK_REGION",
                    env_values,
                    str(state.get("bedrock_region") or CONFIG.online_llm.bedrock_region),
                ),
                bedrock_model_id=self._value(
                    "SUPPORTROLE_BEDROCK_MODEL_ID",
                    env_values,
                    str(
                        state.get("bedrock_model_id")
                        or CONFIG.online_llm.bedrock_model_id
                    ),
                ),
                bedrock_api_key=bedrock_api_key,
            )

    def set_source(self, source: LLMSource) -> None:
        source = self._clean_source(source)
        with self._lock:
            state = self._read_runtime_config()
            state["source"] = source
            self._write_runtime_config(state)

    def store_bedrock_api_key(self, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            return
        with self._lock:
            self._write_env_values({BEDROCK_API_KEY_ENV: key})
            os.environ[BEDROCK_API_KEY_ENV] = key

    def _read_runtime_config(self) -> dict[str, object]:
        if not self.runtime_config_path.exists():
            return {}
        try:
            data = json.loads(self.runtime_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_runtime_config(self, state: dict[str, object]) -> None:
        self.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_config_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_env(self) -> dict[str, str]:
        if not self.env_path.exists():
            return {}
        values: dict[str, str] = {}
        try:
            lines = self.env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return values
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = self._unquote_env_value(value.strip())
        return values

    def _write_env_values(self, updates: dict[str, str]) -> None:
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str]
        if self.env_path.exists():
            lines = self.env_path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                out.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={self._quote_env_value(updates[key])}")
                seen.add(key)
            else:
                out.append(line)

        for key, value in updates.items():
            if key not in seen:
                out.append(f"{key}={self._quote_env_value(value)}")

        self.env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        try:
            os.chmod(self.env_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    @staticmethod
    def _secret(key: str, env_values: dict[str, str]) -> str:
        return (os.environ.get(key) or env_values.get(key) or "").strip()

    @staticmethod
    def _value(key: str, env_values: dict[str, str], default: str) -> str:
        return (os.environ.get(key) or env_values.get(key) or default).strip()

    @staticmethod
    def _clean_source(value: str) -> LLMSource:
        return ONLINE_SOURCE if value.lower() == ONLINE_SOURCE else LOCAL_SOURCE

    @staticmethod
    def _quote_env_value(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @staticmethod
    def _unquote_env_value(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if len(value) >= 2 and value[0] == value[-1] == "'":
            return value[1:-1]
        return value


llm_settings = LLMSettingsStore()

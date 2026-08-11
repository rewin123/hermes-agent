"""Route audio inputs to the model natively, or fall back to transcription.

This is the audio twin of :mod:`agent.image_routing`. The image side has been
merged for a while; the audio side has not, so audio attachments are always
transcribed to text before the model ever sees them, even when the active
model can take audio natively. ``ModelCapabilities.supports_audio_input()``
already exists in :mod:`agent.models_dev` but nothing consumes it — it only
feeds the capability badge.

Two modes, mirroring image routing:

``native``
    Pass the audio through as OpenAI-style ``input_audio`` content parts. The
    model hears the recording: tone, hesitation, background, and — for speech
    LLMs that emit audio — it can answer in the same modality.

``text``
    Pre-transcribe with the existing STT pipeline and send the transcript.
    This is today's behaviour and stays the default for text-only models.

``auto`` (the default) picks ``native`` when the model declares audio input
support, otherwise ``text``. Local and custom models are usually absent from
models.dev, so a ``supports_audio`` override in config.yaml is honoured first
— without it, self-hosted multimodal models could never reach ``native``.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_MODES = {"auto", "native", "text"}

# Kept in sync with tools/transcription_tools.SUPPORTED_FORMATS so that a file
# accepted for transcription is also accepted for native routing; a format
# gap between the two paths would make routing change what is *supported*,
# not just how it is delivered.
SUPPORTED_AUDIO_FORMATS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm",
    ".ogg", ".oga", ".opus", ".aac", ".flac", ".caf",
}

# OpenAI's input_audio part carries a bare format token, not a MIME type.
_FORMAT_TOKENS = {
    ".mp3": "mp3", ".mpga": "mp3", ".mpeg": "mp3",
    ".wav": "wav", ".caf": "wav",
    ".ogg": "ogg", ".oga": "ogg", ".opus": "opus",
    ".webm": "webm", ".m4a": "m4a", ".mp4": "mp4",
    ".aac": "aac", ".flac": "flac",
}

# Same ceiling the transcription path enforces.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    return val if val in _VALID_MODES else "auto"


def _supports_audio_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Resolve user-declared audio capability from config.yaml.

    Resolution order, first hit wins:
      1. ``model.supports_audio``
      2. ``providers.<provider>.models.<model>.supports_audio`` (or the
         shorter ``audio`` alias)

    Returns None when nothing is declared, so the caller falls through to
    models.dev. Mirrors ``image_routing._supports_vision_override``; named
    custom providers are canonicalized to ``custom`` at runtime, so the
    user-declared name under ``model.provider`` is tried as well.
    """
    if not isinstance(cfg, dict):
        return None

    try:
        from agent.image_routing import _coerce_capability_bool
    except Exception:  # pragma: no cover - defensive
        def _coerce_capability_bool(raw: Any) -> Optional[bool]:  # type: ignore
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                v = raw.strip().lower()
                if v in {"true", "yes", "1", "on"}:
                    return True
                if v in {"false", "no", "0", "off"}:
                    return False
            return None

    model_cfg_raw = cfg.get("model")
    model_cfg: Dict[str, Any] = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}

    top = _coerce_capability_bool(model_cfg.get("supports_audio"))
    if top is not None:
        return top

    config_provider = str(model_cfg.get("provider") or "").strip()
    candidates: List[str] = []
    for candidate in (requested_provider, provider, config_provider):
        if not candidate:
            continue
        candidates.append(candidate)
        if candidate.startswith("custom:"):
            stripped = candidate[len("custom:"):]
            if stripped:
                candidates.append(stripped)

    providers_cfg = cfg.get("providers")
    if not isinstance(providers_cfg, dict):
        return None

    target_model = str(model or "").strip()
    for cand in candidates:
        entry = providers_cfg.get(cand)
        if not isinstance(entry, dict):
            continue
        models_cfg = entry.get("models")
        if not isinstance(models_cfg, dict):
            continue
        per_model = models_cfg.get(target_model)
        if not isinstance(per_model, dict):
            continue
        for key in ("supports_audio", "audio"):
            val = _coerce_capability_bool(per_model.get(key))
            if val is not None:
                return val
    return None


def _lookup_supports_audio(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """True/False when resolvable, None when unknown."""
    override = _supports_audio_override(
        cfg, provider, model, requested_provider=requested_provider
    )
    if override is not None:
        return override
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "audio_routing: caps lookup failed for %s:%s — %s", provider, model, exc
        )
        return None
    if caps is None:
        return None
    return bool(caps.supports_audio_input())


def decide_audio_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
    *,
    requested_provider: str = "",
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn."""
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("audio_input_mode"))

    if mode_cfg in ("native", "text"):
        return mode_cfg

    supports = _lookup_supports_audio(
        provider, model, cfg, requested_provider=requested_provider
    )
    return "native" if supports is True else "text"


def format_token(path: Path) -> Optional[str]:
    """Extension -> the token OpenAI expects in ``input_audio.format``."""
    return _FORMAT_TOKENS.get(path.suffix.lower())


def build_native_audio_parts(
    text: str,
    audio_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build OpenAI-style content parts, returning ``(parts, skipped)``.

    Unreadable, oversized and unsupported files are reported in ``skipped``
    rather than raising, so the caller can fall back to transcription for
    those and still send the rest — a single bad attachment must not lose
    the whole turn.
    """
    parts: List[Dict[str, Any]] = []
    skipped: List[str] = []

    if text:
        parts.append({"type": "text", "text": text})

    for raw_path in audio_paths:
        path = Path(raw_path)
        token = format_token(path)
        if token is None or path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            skipped.append(raw_path)
            continue
        try:
            if path.stat().st_size > MAX_AUDIO_BYTES:
                skipped.append(raw_path)
                continue
            data = path.read_bytes()
        except OSError as exc:
            logger.debug("audio_routing: unreadable audio %s — %s", raw_path, exc)
            skipped.append(raw_path)
            continue
        parts.append({
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(data).decode("ascii"),
                "format": token,
            },
        })

    return parts, skipped


def has_native_audio_parts(content: Any) -> bool:
    """True when a message content list carries at least one audio part."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(p, dict) and p.get("type") in ("input_audio", "audio")
        for p in content
    )

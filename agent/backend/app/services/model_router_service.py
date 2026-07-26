"""
Runtime model router for distilled role models.

The OPD pipeline can register checkpoints, but AdGenie can only call a model
when it is exposed through an OpenAI-compatible endpoint. This service reads
storage/model_router.json and resolves a role to provider/base_url/model/API key
overrides for the LLM factory.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
ROUTER_PATH = STORAGE_DIR / "model_router.json"

VALID_ROLES = {
    "agent_orchestration",
    "video_script",
    "image_understanding",
    "tts_voice",
    "personal_agent",
}

# Roles with an actual runtime call site today. The remaining roles are
# placeholders until their tools/subsystems call create_llm(role=...) or resolve
# the router explicitly.
RUNTIME_CONNECTED_ROLES = {"agent_orchestration", "image_understanding", "tts_voice", "video_script", "personal_agent"}


def _role_default(role: str) -> dict[str, Any]:
    entry = {
        "enabled": False,
        "provider": "",
        "model": "",
        "base_url": "",
        "api_key_env": "",
        "checkpoint": "",
        "runtime_connected": role in RUNTIME_CONNECTED_ROLES,
    }
    if role == "agent_orchestration":
        entry.update({"fallback_provider": "", "fallback_model": ""})
    return entry


DEFAULT_ROUTER = {
    "enabled": False,
    "default": {
        "provider": "",
        "model": "",
        "base_url": "",
        "api_key_env": "",
    },
    "models": {role: _role_default(role) for role in sorted(VALID_ROLES)},
}


@dataclass
class RoutedModel:
    role: str
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    source: str = "default"


def _normalize_router(data: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(data) if isinstance(data, dict) else {}
    normalized.setdefault("enabled", False)

    default = normalized.get("default")
    if not isinstance(default, dict):
        normalized["default"] = {
            "provider": "",
            "model": default or "",
            "base_url": "",
            "api_key_env": "",
        }
    else:
        for key in ("provider", "model", "base_url", "api_key_env"):
            default.setdefault(key, "")

    models = normalized.setdefault("models", {})
    for role in VALID_ROLES:
        existing = models.get(role)
        if not isinstance(existing, dict):
            existing = {}
        merged = _role_default(role)
        merged.update(existing)
        merged["runtime_connected"] = role in RUNTIME_CONNECTED_ROLES
        models[role] = merged
    return normalized


def ensure_model_router_defaults() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not ROUTER_PATH.exists():
        ROUTER_PATH.write_text(
            json.dumps(DEFAULT_ROUTER, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Created model router config: {ROUTER_PATH}")


def load_model_router() -> dict[str, Any]:
    ensure_model_router_defaults()
    try:
        raw = json.loads(ROUTER_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to load model router config, using defaults: {e}")
        raw = DEFAULT_ROUTER
    return _normalize_router(raw)


def save_model_router(data: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ROUTER_PATH.write_text(
        json.dumps(_normalize_router(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_model_for_role(role: Optional[str]) -> Optional[RoutedModel]:
    """
    Resolve a role to a callable OpenAI-compatible model.

    Returns None when routing is disabled, role config is disabled, the role has
    no runtime call site, or the role only contains an offline checkpoint path
    without endpoint information.
    """
    if not role:
        return None

    if role not in RUNTIME_CONNECTED_ROLES:
        logger.warning(
            "Unknown model router role: %s",
            role,
        )
        return None

    config = load_model_router()
    if not config.get("enabled", False):
        return None

    entry = (config.get("models") or {}).get(role)
    if not isinstance(entry, dict) or not entry.get("enabled", False):
        return None

    default_config = config.get("default") if isinstance(config.get("default"), dict) else {}
    provider = (entry.get("provider") or default_config.get("provider") or "").strip().lower()
    model = (
        entry.get("model")
        or entry.get("model_name")
        or entry.get("primary")
        or default_config.get("model")
        or ""
    ).strip()
    base_url = (entry.get("base_url") or default_config.get("base_url") or "").strip()
    api_key_env = (entry.get("api_key_env") or default_config.get("api_key_env") or "").strip()
    api_key = (entry.get("api_key") or "").strip()
    if api_key_env and not api_key:
        api_key = os.getenv(api_key_env, "").strip()

    checkpoint = (entry.get("checkpoint") or "").strip()
    if not provider or not model:
        if checkpoint:
            logger.warning(
                "Role '%s' has checkpoint '%s' but no callable provider/model. "
                "Serve the distilled checkpoint through an OpenAI-compatible endpoint "
                "and fill provider/model/base_url/api_key_env in model_router.json.",
                role,
                checkpoint,
            )
        return None

    if provider not in ("volcano", "siliconflow", "openai_compatible"):
        logger.warning("Unsupported routed provider for role '%s': %s", role, provider)
        return None

    if provider == "openai_compatible" and not api_key:
        api_key = "EMPTY"

    return RoutedModel(
        role=role,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        source="model_router.json",
    )

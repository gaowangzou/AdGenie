"""
Configure AdGenie runtime model routing from the command line.

Examples:
  py scripts/configure_model_router.py show
  py scripts/configure_model_router.py enable --role agent_orchestration \
    --provider openai_compatible --model agent_orchestration-test \
    --base-url http://127.0.0.1:30000/v1 --check
  py scripts/configure_model_router.py disable --role agent_orchestration
"""
from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROUTER_PATH = BACKEND_DIR / "storage" / "model_router.json"
RUNTIME_CONNECTED_ROLES = {"agent_orchestration", "image_understanding", "tts_voice", "video_script", "personal_agent"}
VALID_ROLES = {
    "agent_orchestration",
    "video_script",
    "image_understanding",
    "tts_voice",
    "personal_agent",
}


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


def normalize_router(data: dict[str, Any]) -> dict[str, Any]:
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


def load_router() -> dict[str, Any]:
    if not ROUTER_PATH.exists():
        ROUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_router(DEFAULT_ROUTER)
    try:
        data = json.loads(ROUTER_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = DEFAULT_ROUTER
    return normalize_router(data)


def save_router(data: dict[str, Any]) -> None:
    ROUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROUTER_PATH.write_text(json.dumps(normalize_router(data), ensure_ascii=False, indent=2), encoding="utf-8")


def check_openai_endpoint(base_url: str, model: str) -> None:
    base_url = base_url.rstrip("/")
    if not base_url:
        raise SystemExit("--base-url is required for --check")

    req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise SystemExit(f"Endpoint check failed: {e}") from e

    try:
        payload = json.loads(body)
        model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    except Exception:
        model_ids = []

    if model_ids and model not in model_ids:
        print(f"Warning: endpoint responded, but model '{model}' was not listed.")
        print(f"Available models: {', '.join(str(m) for m in model_ids)}")
    else:
        print(f"Endpoint check ok: {base_url}")


def enable_role(args: argparse.Namespace) -> None:
    if args.role not in VALID_ROLES:
        raise SystemExit(f"Unknown role: {args.role}. Valid roles: {', '.join(sorted(VALID_ROLES))}")

    if not args.provider or not args.model:
        raise SystemExit("--provider and --model are required for enable")
    if args.provider == "openai_compatible" and not args.base_url:
        raise SystemExit("--base-url is required for provider=openai_compatible")

    data = load_router()
    data["enabled"] = True
    entry = data["models"].setdefault(args.role, _role_default(args.role))
    entry.update({
        "enabled": True,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url or "",
        "api_key_env": args.api_key_env or "",
        "checkpoint": args.checkpoint or entry.get("checkpoint", ""),
        "runtime_connected": args.role in RUNTIME_CONNECTED_ROLES,
    })
    save_router(data)

    if args.check:
        check_openai_endpoint(entry["base_url"], entry["model"])

    print(f"Enabled role route: {args.role} -> {entry['provider']}:{entry['model']}")
    print(f"Config: {ROUTER_PATH}")


def disable_role(args: argparse.Namespace) -> None:
    data = load_router()
    if args.role:
        if args.role not in VALID_ROLES:
            raise SystemExit(f"Unknown role: {args.role}. Valid roles: {', '.join(sorted(VALID_ROLES))}")
        data["models"].setdefault(args.role, _role_default(args.role))["enabled"] = False
        print(f"Disabled role route: {args.role}")
    else:
        data["enabled"] = False
        print("Disabled model router globally")
    save_router(data)


def show(_: argparse.Namespace) -> None:
    print(json.dumps(load_router(), ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure AdGenie model_router.json")
    sub = parser.add_subparsers(dest="command", required=True)

    enable = sub.add_parser("enable", help="Enable a role route")
    enable.add_argument("--role", required=True)
    enable.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "siliconflow", "volcano"])
    enable.add_argument("--model", required=True)
    enable.add_argument("--base-url", default="")
    enable.add_argument("--api-key-env", default="")
    enable.add_argument("--checkpoint", default="")
    enable.add_argument("--check", action="store_true")
    enable.set_defaults(func=enable_role)

    disable = sub.add_parser("disable", help="Disable one role route or the global router")
    disable.add_argument("--role", default="")
    disable.set_defaults(func=disable_role)

    check = sub.add_parser("check", help="Check an OpenAI-compatible endpoint")
    check.add_argument("--base-url", required=True)
    check.add_argument("--model", required=True)
    check.set_defaults(func=lambda args: check_openai_endpoint(args.base_url, args.model))

    show_cmd = sub.add_parser("show", help="Print current model router config")
    show_cmd.set_defaults(func=show)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

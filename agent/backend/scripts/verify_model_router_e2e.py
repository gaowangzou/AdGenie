"""
Run an end-to-end smoke test for AdGenie model routing against a running backend.

This script is intended for real small-model validation after a local vLLM/SGLang
OpenAI-compatible endpoint is already running. It temporarily writes
storage/model_router.json, calls /api/chat, prints SSE events, and restores the
previous router config by default.

Examples:
  py scripts/verify_model_router_e2e.py \
    --backend-url http://127.0.0.1:8000 \
    --role agent_orchestration \
    --model agent_orchestration-test \
    --base-url http://127.0.0.1:30000/v1 \
    --prompt "请调用 list_skill_dir 查看 custom 目录，然后用一句话总结目录内容" \
    --expect-tool list_skill_dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROUTER_PATH = BACKEND_DIR / "storage" / "model_router.json"
RUNTIME_CONNECTED_ROLES = {"agent_orchestration", "image_understanding", "tts_voice", "video_script", "personal_agent"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_route(args: argparse.Namespace) -> dict[str, Any]:
    previous = load_json(ROUTER_PATH)
    data = json.loads(json.dumps(previous)) if previous else {"enabled": False, "default": {}, "models": {}}
    data.setdefault("default", {"provider": "", "model": "", "base_url": "", "api_key_env": ""})
    data.setdefault("models", {})
    data["enabled"] = True
    entry = data["models"].setdefault(args.role, {})
    entry.update({
        "enabled": True,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "checkpoint": entry.get("checkpoint", ""),
        "runtime_connected": args.role in RUNTIME_CONNECTED_ROLES,
    })
    write_json(ROUTER_PATH, data)
    return previous


def restore_route(previous: dict[str, Any]) -> None:
    if previous:
        write_json(ROUTER_PATH, previous)


def request_json(url: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_model_endpoint(base_url: str, model: str) -> None:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    if ids and model not in ids:
        raise RuntimeError(f"model '{model}' not listed by endpoint. available={ids}")


def parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):].strip()
        if not body or body == "[DONE]":
            continue
        try:
            events.append(json.loads(body))
        except Exception:
            events.append({"type": "raw", "content": body})
    return events


def call_chat(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    payload = {
        "message": args.prompt,
        "session_id": f"router-e2e-{int(time.time())}",
        "canvas_id": args.canvas_id,
    }
    req = urllib.request.Request(
        f"{args.backend_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return raw, parse_sse(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AdGenie model router against a running backend")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--role", default="agent_orchestration", choices=sorted(RUNTIME_CONNECTED_ROLES))
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "siliconflow", "volcano"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--prompt", default="请调用 list_skill_dir 查看 custom 目录，然后用一句话总结目录内容")
    parser.add_argument("--expect-text", default="")
    parser.add_argument("--expect-tool", default="")
    parser.add_argument("--canvas-id", default="router-e2e-smoke")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--keep-router", action="store_true", help="Do not restore model_router.json after the test")
    args = parser.parse_args()

    previous: dict[str, Any] = {}
    try:
        health = request_json(f"{args.backend_url.rstrip('/')}/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"backend health check failed: {health}")
        check_model_endpoint(args.base_url, args.model)
        previous = configure_route(args)
        raw, events = call_chat(args)

        print("=== Raw SSE ===")
        print(raw)
        print("=== Summary ===")
        text = "".join(str(ev.get("content", "")) for ev in events if ev.get("type") == "delta")
        tools = [ev.get("name") for ev in events if ev.get("type") == "tool_call"]
        errors = [ev for ev in events if ev.get("type") == "error"]
        print(json.dumps({"text": text, "tools": tools, "errors": errors}, ensure_ascii=False, indent=2))

        if args.expect_text and args.expect_text not in text:
            raise RuntimeError(f"expected text not found: {args.expect_text!r}")
        if args.expect_tool and args.expect_tool not in tools:
            raise RuntimeError(f"expected tool_call not found: {args.expect_tool!r}; tools={tools}")
        if errors:
            raise RuntimeError(f"chat returned error events: {errors}")
        return 0
    except (urllib.error.URLError, RuntimeError) as e:
        print(f"VERIFY_FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_router:
            restore_route(previous)


if __name__ == "__main__":
    raise SystemExit(main())

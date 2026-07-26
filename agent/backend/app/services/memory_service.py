"""
Structured memory service for AdGenie.

Tier 1: long-term memory records stored in SQLite.
Tier 2: per-canvas session digests for context compression.

MEMORY.md is kept as a human-readable mirror, not the source of truth.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
WORKSPACE_DIR = BASE_DIR / "workspace"
DB_PATH = STORAGE_DIR / "memory.sqlite3"
MEMORY_MD_PATH = WORKSPACE_DIR / "MEMORY.md"

EMBED_DIM = 128
SECTION_TO_TYPE = {
    "角色资产": "角色资产",
    "成功 Prompt 模板": "prompt模板",
    "用户偏好记录": "用户偏好",
}
CONFIDENCE_ORDER = {
    "rejected": 0,
    "deprecated": 1,
    "inferred": 2,
    "confirmed": 3,
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _tokens(text: str) -> Iterable[str]:
    """Small local tokenizer for mixed Chinese/English text."""
    lowered = text.lower()
    for token in re.findall(r"[a-z0-9_]+", lowered):
        yield token
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    for ch in cjk:
        yield ch
    for i in range(max(len(cjk) - 1, 0)):
        yield "".join(cjk[i:i + 2])


def embed_text(text: str) -> list[float]:
    """
    Deterministic local embedding.

    This avoids adding a vector database or external embedding dependency. It is
    good enough for hundreds to low-thousands of project memories and can be
    replaced later by a provider-backed embedding model.
    """
    vec = [0.0] * EMBED_DIM
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % EMBED_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def init_memory_store() -> None:
    """Create SQLite tables and migrate legacy MEMORY.md content once."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                scope_type TEXT NOT NULL DEFAULT 'user',
                scope_id TEXT NOT NULL DEFAULT 'default_user',
                confidence TEXT NOT NULL DEFAULT 'inferred',
                source_session_id TEXT NOT NULL DEFAULT '',
                source_message_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL DEFAULT '',
                use_count INTEGER NOT NULL DEFAULT 0,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_digests (
                id TEXT PRIMARY KEY,
                canvas_id TEXT NOT NULL UNIQUE,
                digest TEXT NOT NULL,
                task_type TEXT NOT NULL DEFAULT '',
                final_outputs_json TEXT NOT NULL DEFAULT '[]',
                satisfaction TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_records(scope_type, scope_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_records(type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_digest_canvas ON session_digests(canvas_id)"
        )
        conn.commit()

    _migrate_memory_md_once()


def _content_hash(type_: str, content: str, scope_type: str, scope_id: str) -> str:
    raw = "\n".join([type_, content.strip(), scope_type, scope_id])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def upsert_memory(
    *,
    type: str,
    content: str,
    summary: str = "",
    scope_type: str = "user",
    scope_id: str = "default_user",
    confidence: str = "inferred",
    source_session_id: str = "",
    source_message_id: str = "",
    metadata: Optional[dict[str, Any]] = None,
    refresh_mirror: bool = True,
) -> str:
    """Insert a memory record, or upgrade an existing duplicate."""
    init_memory_store()
    normalized = content.strip()
    if not normalized:
        raise ValueError("memory content cannot be empty")
    if confidence not in CONFIDENCE_ORDER:
        confidence = "inferred"

    now = _now()
    record_id = f"mem_{uuid.uuid4().hex[:12]}"
    summary_text = summary.strip() or normalized[:160]
    content_hash = _content_hash(type, normalized, scope_type, scope_id)
    embedding = json.dumps(embed_text(f"{type}\n{summary_text}\n{normalized}"))
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, confidence, use_count FROM memory_records WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            best_confidence = max(
                existing["confidence"],
                confidence,
                key=lambda c: CONFIDENCE_ORDER.get(c, 0),
            )
            conn.execute(
                """
                UPDATE memory_records
                SET confidence = ?, summary = ?, updated_at = ?, source_session_id = ?,
                    source_message_id = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    best_confidence,
                    summary_text,
                    now,
                    source_session_id,
                    source_message_id,
                    metadata_json,
                    existing["id"],
                ),
            )
            conn.commit()
            if refresh_mirror:
                export_memory_markdown()
            return existing["id"]

        conn.execute(
            """
            INSERT INTO memory_records (
                id, type, content, summary, scope_type, scope_id, confidence,
                source_session_id, source_message_id, created_at, updated_at,
                embedding_json, metadata_json, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                type,
                normalized,
                summary_text,
                scope_type,
                scope_id,
                confidence,
                source_session_id,
                source_message_id,
                now,
                now,
                embedding,
                metadata_json,
                content_hash,
            ),
        )
        conn.commit()

    if refresh_mirror:
        export_memory_markdown()
    return record_id


def retrieve_relevant_memories(
    query: str,
    *,
    scope_id: str = "default_user",
    canvas_id: Optional[str] = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Retrieve Tier 1 memories relevant to the current request."""
    init_memory_store()
    query = (query or "").strip()
    if not query:
        return []
    query_vec = embed_text(query)
    rows: list[sqlite3.Row]
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_records
            WHERE confidence NOT IN ('rejected', 'deprecated')
              AND (scope_type != 'canvas' OR scope_id = ?)
              AND (scope_type != 'user' OR scope_id IN (?, 'default_user'))
            """,
            (canvas_id or "", scope_id),
        ).fetchall()

    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        try:
            memory_vec = json.loads(row["embedding_json"])
        except Exception:
            memory_vec = []
        score = _cosine(query_vec, memory_vec)
        if row["confidence"] == "confirmed":
            score += 0.12
        if row["scope_id"] == scope_id:
            score += 0.08
        if canvas_id and row["scope_id"] == canvas_id:
            score += 0.18
        score += min(row["use_count"], 5) * 0.01
        min_score = 0.34 if row["type"] == "prompt模板" else 0.12
        if score >= min_score:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:limit]
    if selected:
        with _connect() as conn:
            for _, row in selected:
                conn.execute(
                    """
                    UPDATE memory_records
                    SET last_used_at = ?, use_count = use_count + 1
                    WHERE id = ?
                    """,
                    (_now(), row["id"]),
                )
            conn.commit()

    return [_row_to_memory_dict(row, score) for score, row in selected]


def _row_to_memory_dict(row: sqlite3.Row, score: float = 0.0) -> dict[str, Any]:
    return {
        "id": row["id"],
        "type": row["type"],
        "content": row["content"],
        "summary": row["summary"],
        "scope_type": row["scope_type"],
        "scope_id": row["scope_id"],
        "confidence": row["confidence"],
        "source_session_id": row["source_session_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "use_count": row["use_count"],
        "score": round(score, 4),
    }


def get_session_digest(canvas_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not canvas_id:
        return None
    init_memory_store()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_digests WHERE canvas_id = ?",
            (canvas_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "canvas_id": row["canvas_id"],
        "digest": row["digest"],
        "task_type": row["task_type"],
        "final_outputs": json.loads(row["final_outputs_json"] or "[]"),
        "satisfaction": row["satisfaction"],
        "updated_at": row["updated_at"],
    }


def upsert_session_digest(
    *,
    canvas_id: str,
    user_message: str,
    assistant_content: str,
    tool_results: list[dict[str, Any]],
    previous_messages: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Update a compact digest for the current canvas after each turn."""
    if not canvas_id:
        return ""
    init_memory_store()
    now = _now()
    existing = get_session_digest(canvas_id)
    task_type = _infer_task_type(user_message, tool_results)
    outputs = _extract_outputs(tool_results)
    satisfaction = _infer_satisfaction(user_message, assistant_content)

    prior = existing["digest"] if existing else ""
    recent_user_messages = [
        m.get("content", "") for m in (previous_messages or [])[-6:]
        if m.get("role") == "user" and m.get("content")
    ]
    digest_parts = []
    if prior:
        digest_parts.append(prior)
    if recent_user_messages:
        digest_parts.append("近期用户需求：" + " / ".join(recent_user_messages[-3:]))
    digest_parts.append(f"本轮用户需求：{user_message.strip()[:300]}")
    if assistant_content.strip():
        digest_parts.append(f"本轮结果：{assistant_content.strip()[:300]}")
    if outputs:
        digest_parts.append(
            "最终产物：" + ", ".join(
                f"{item.get('type', 'asset')} {item.get('url', '')}" for item in outputs[:5]
            )
        )
    digest = "\n".join(digest_parts)
    digest = _compact_digest(digest)
    embedding = json.dumps(embed_text(f"{task_type}\n{digest}"))
    digest_id = existing["id"] if existing else f"digest_{uuid.uuid4().hex[:12]}"

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO session_digests (
                id, canvas_id, digest, task_type, final_outputs_json,
                satisfaction, created_at, updated_at, embedding_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canvas_id) DO UPDATE SET
                digest = excluded.digest,
                task_type = excluded.task_type,
                final_outputs_json = excluded.final_outputs_json,
                satisfaction = excluded.satisfaction,
                updated_at = excluded.updated_at,
                embedding_json = excluded.embedding_json,
                metadata_json = excluded.metadata_json
            """,
            (
                digest_id,
                canvas_id,
                digest,
                task_type,
                json.dumps(outputs, ensure_ascii=False),
                satisfaction,
                existing["updated_at"] if existing else now,
                now,
                embedding,
                json.dumps({"source": "chat_turn"}, ensure_ascii=False),
            ),
        )
        conn.commit()
    return digest_id


def extract_inferred_memories_from_turn(
    *,
    canvas_id: str,
    user_message: str,
    assistant_content: str,
    tool_results: list[dict[str, Any]],
) -> list[str]:
    """
    Heuristic first-pass memory extraction.

    Confirmed memories still come from write_memory. These inferred records are
    deliberately conservative and clearly labelled as inferred.
    """
    init_memory_store()
    created: list[str] = []
    text = user_message.strip()
    if not text:
        return created

    preference_markers = ("喜欢", "偏好", "以后", "默认", "希望以后", "下次")
    negative_markers = ("不喜欢", "不要", "避免", "别用", "不想要")
    confirm_markers = ("满意", "可以", "不错", "就这个", "确认", "很好", "挺好")
    outputs = _extract_outputs(tool_results)

    if any(marker in text for marker in negative_markers):
        created.append(upsert_memory(
            type="用户偏好",
            content=f"用户要求避免或不偏好：{text}",
            summary=text[:160],
            scope_type="user",
            scope_id="default_user",
            confidence="inferred",
            source_session_id=canvas_id,
            metadata={"polarity": "negative", "extractor": "heuristic"},
        ))
    elif any(marker in text for marker in preference_markers):
        created.append(upsert_memory(
            type="用户偏好",
            content=f"用户表达了偏好或默认要求：{text}",
            summary=text[:160],
            scope_type="user",
            scope_id="default_user",
            confidence="inferred",
            source_session_id=canvas_id,
            metadata={"polarity": "positive", "extractor": "heuristic"},
        ))

    if outputs and any(marker in text for marker in confirm_markers):
        created.append(upsert_memory(
            type="角色资产" if "角色" in text or "形象" in text else "用户偏好",
            content=f"用户对当前产物给出正面确认：{text}；相关产物：{json.dumps(outputs[:3], ensure_ascii=False)}",
            summary=f"用户确认当前产物可用：{text[:120]}",
            scope_type="canvas",
            scope_id=canvas_id or "default_user",
            confidence="inferred",
            source_session_id=canvas_id,
            metadata={"outputs": outputs[:3], "extractor": "heuristic"},
        ))

    return created


def build_retrieved_memory_context(query: str, canvas_id: Optional[str] = None) -> str:
    """Build prompt-ready Tier 1 and Tier 2 context."""
    memories = retrieve_relevant_memories(query, canvas_id=canvas_id)
    digest = get_session_digest(canvas_id)
    sections: list[str] = []

    if digest:
        sections.append(
            "<session_digest>\n"
            f"任务类型：{digest['task_type'] or 'unknown'}\n"
            f"满意度：{digest['satisfaction']}\n"
            f"摘要：{digest['digest']}\n"
            "</session_digest>"
        )

    if memories:
        lines = []
        for memory in memories:
            lines.append(
                f"- [{memory['confidence']}][{memory['type']}] {memory['summary'] or memory['content']}"
            )
        sections.append("<relevant_memory>\n" + "\n".join(lines) + "\n</relevant_memory>")

    return "\n".join(sections)


def export_memory_markdown() -> None:
    """Render structured records to MEMORY.md for human review/editing."""
    init_memory_store_without_migration()
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM memory_records
            WHERE confidence NOT IN ('rejected', 'deprecated')
            ORDER BY
              CASE confidence
                WHEN 'confirmed' THEN 0
                WHEN 'inferred' THEN 1
                ELSE 2
              END,
              updated_at DESC
            """
        ).fetchall()

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["type"], []).append(row)

    lines = [
        "# 创作记忆",
        "",
        "> 这是结构化长期记忆的可读镜像。Agent 内部以 storage/memory.sqlite3 为准。",
        "> 可以在设置页查看，但自动写入会在下一次镜像刷新时重排本文件。",
        "",
    ]
    for type_name in ("角色资产", "prompt模板", "用户偏好"):
        section_title = "成功 Prompt 模板" if type_name == "prompt模板" else (
            "用户偏好记录" if type_name == "用户偏好" else type_name
        )
        lines.append(f"## {section_title}")
        items = grouped.get(type_name, [])
        if not items:
            lines.append("（暂无结构化记录）")
        for row in items:
            lines.append(
                f"- `{row['id']}` [{row['confidence']}] {row['content']}"
                f"（scope: {row['scope_type']}/{row['scope_id']}，updated: {row['updated_at']}）"
            )
        lines.append("")

    MEMORY_MD_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def init_memory_store_without_migration() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                scope_type TEXT NOT NULL DEFAULT 'user',
                scope_id TEXT NOT NULL DEFAULT 'default_user',
                confidence TEXT NOT NULL DEFAULT 'inferred',
                source_session_id TEXT NOT NULL DEFAULT '',
                source_message_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL DEFAULT '',
                use_count INTEGER NOT NULL DEFAULT 0,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.commit()


def import_memory_markdown(content: str, source_session_id: str = "manual_memory_md") -> list[str]:
    """
    Sync manually edited MEMORY.md content into structured memory.

    Existing mirror rows keep their visible confidence (`[inferred]` stays
    inferred). New hand-written rows without a mirror prefix are treated as
    explicit manual memories and become confirmed. Active records removed from
    the mirror are marked deprecated instead of being resurrected on save.
    """
    init_memory_store_without_migration()
    imported: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    processed_types: set[str] = set()

    for section, type_name in SECTION_TO_TYPE.items():
        if f"## {section}" not in content:
            continue
        processed_types.add(type_name)
        body = _extract_markdown_section(content, section)
        for entry in _split_legacy_entries(body):
            parsed = _parse_imported_entry(entry)
            plain = parsed["content"]
            if not plain:
                continue
            scope_type = parsed["scope_type"]
            scope_id = parsed["scope_id"]
            seen_hashes.add(_content_hash(type_name, plain, scope_type, scope_id))
            memory_id = _upsert_imported_memory(
                record_id=parsed["id"],
                type=type_name,
                content=plain,
                confidence=parsed["confidence"],
                scope_type=scope_type,
                scope_id=scope_id,
                source_session_id=source_session_id,
                section=section,
            )
            imported.append(memory_id)
            seen_ids.add(memory_id)

    _mark_missing_markdown_records_deprecated(
        processed_types=processed_types,
        seen_ids=seen_ids,
        seen_hashes=seen_hashes,
        source_session_id=source_session_id,
    )
    return imported


def _upsert_imported_memory(
    *,
    record_id: str,
    type: str,
    content: str,
    confidence: str,
    scope_type: str,
    scope_id: str,
    source_session_id: str,
    section: str,
) -> str:
    if confidence not in CONFIDENCE_ORDER:
        confidence = "confirmed"
    normalized = content.strip()
    summary = normalized[:160]
    now = _now()
    embedding = json.dumps(embed_text(f"{type}\n{summary}\n{normalized}"))
    metadata_json = json.dumps(
        {"source": "MEMORY.md manual sync", "section": section},
        ensure_ascii=False,
    )
    content_hash = _content_hash(type, normalized, scope_type, scope_id)

    if record_id:
        with _connect() as conn:
            existing = conn.execute(
                "SELECT id FROM memory_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if existing:
                try:
                    conn.execute(
                        """
                        UPDATE memory_records
                        SET type = ?, content = ?, summary = ?, scope_type = ?, scope_id = ?,
                            confidence = ?, source_session_id = ?, updated_at = ?,
                            embedding_json = ?, metadata_json = ?, content_hash = ?
                        WHERE id = ?
                        """,
                        (
                            type,
                            normalized,
                            summary,
                            scope_type,
                            scope_id,
                            confidence,
                            source_session_id,
                            now,
                            embedding,
                            metadata_json,
                            content_hash,
                            record_id,
                        ),
                    )
                    conn.commit()
                    return record_id
                except sqlite3.IntegrityError:
                    logger.warning(
                        "Manual MEMORY.md sync hit duplicate content_hash for %s; falling back to upsert",
                        record_id,
                    )

    return upsert_memory(
        type=type,
        content=normalized,
        summary=summary,
        scope_type=scope_type,
        scope_id=scope_id,
        confidence=confidence,
        source_session_id=source_session_id,
        metadata={"source": "MEMORY.md manual sync", "section": section},
        refresh_mirror=False,
    )


def _mark_missing_markdown_records_deprecated(
    *,
    processed_types: set[str],
    seen_ids: set[str],
    seen_hashes: set[str],
    source_session_id: str,
) -> None:
    if not processed_types:
        return
    placeholders = ",".join("?" for _ in processed_types)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, content_hash FROM memory_records
            WHERE confidence NOT IN ('rejected', 'deprecated')
              AND type IN ({placeholders})
            """,
            tuple(processed_types),
        ).fetchall()
        missing_ids = [
            row["id"] for row in rows
            if row["id"] not in seen_ids and row["content_hash"] not in seen_hashes
        ]
        if not missing_ids:
            return
        id_placeholders = ",".join("?" for _ in missing_ids)
        conn.execute(
            f"""
            UPDATE memory_records
            SET confidence = 'deprecated', updated_at = ?, source_session_id = ?
            WHERE id IN ({id_placeholders})
            """,
            (_now(), source_session_id, *missing_ids),
        )
        conn.commit()

def _migrate_memory_md_once() -> None:
    marker = STORAGE_DIR / ".memory_md_migrated"
    if marker.exists() or not MEMORY_MD_PATH.exists():
        return
    try:
        content = MEMORY_MD_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read legacy MEMORY.md for migration: {e}")
        return

    # Create the marker before inserting records so upsert_memory() does not
    # recursively trigger this migration.
    marker.write_text(_now(), encoding="utf-8")

    for section, type_name in SECTION_TO_TYPE.items():
        body = _extract_markdown_section(content, section)
        if not body:
            continue
        entries = _split_legacy_entries(body)
        for entry in entries:
            plain = entry.strip()
            if not plain or plain.startswith("（Agent 在此记录"):
                continue
            try:
                upsert_memory(
                    type=type_name,
                    content=plain,
                    summary=plain[:160],
                    confidence="confirmed",
                    source_session_id="legacy_memory_md",
                    metadata={"source": "MEMORY.md migration"},
                )
            except Exception as e:
                logger.warning(f"Failed to migrate memory entry: {e}")

    export_memory_markdown()


def _parse_imported_entry(entry: str) -> dict[str, str]:
    plain = entry.strip()
    result = {
        "id": "",
        "confidence": "confirmed",
        "scope_type": "user",
        "scope_id": "default_user",
        "content": "",
    }

    mirror_match = re.match(
        r"^`(?P<id>mem_[^`]+)`\s+\[(?P<confidence>[^\]]+)\]\s+(?P<body>.*)$",
        plain,
        re.DOTALL,
    )
    if mirror_match:
        result["id"] = mirror_match.group("id")
        visible_confidence = mirror_match.group("confidence").strip()
        if visible_confidence in CONFIDENCE_ORDER:
            result["confidence"] = visible_confidence
        plain = mirror_match.group("body").strip()

    scope_match = re.search(
        r"（scope:\s*(?P<scope_type>[^/）]+)/(?P<scope_id>[^，）]+)，updated:\s*[^）]+）\s*$",
        plain,
        re.DOTALL,
    )
    if scope_match:
        result["scope_type"] = scope_match.group("scope_type").strip() or "user"
        result["scope_id"] = scope_match.group("scope_id").strip() or "default_user"
        plain = plain[:scope_match.start()].strip()

    if plain.startswith("（Agent 在此记录") or plain.startswith("（暂无结构化记录"):
        plain = ""
    result["content"] = plain
    return result

def _extract_markdown_section(content: str, section: str) -> str:
    header = f"## {section}"
    start = content.find(header)
    if start == -1:
        return ""
    body_start = start + len(header)
    next_start = content.find("\n## ", body_start)
    if next_start == -1:
        return content[body_start:].strip()
    return content[body_start:next_start].strip()


def _split_legacy_entries(body: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("（Agent 在此记录"):
            continue
        if stripped.startswith("### ") or stripped.startswith("- "):
            if current:
                chunks.append("\n".join(current).strip())
            current = [stripped.lstrip("- ").strip()]
        else:
            if current:
                current.append(stripped)
            else:
                chunks.append(stripped)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def _infer_task_type(user_message: str, tool_results: list[dict[str, Any]]) -> str:
    text = user_message.lower()
    if "3d" in text or "3D" in user_message or "模型" in user_message:
        return "3d"
    if "视频" in user_message:
        return "video"
    if "音频" in user_message or "播客" in user_message or "tts" in text:
        return "audio"
    if "图片" in user_message or "图" in user_message or "海报" in user_message:
        return "image"
    outputs = _extract_outputs(tool_results)
    return outputs[0]["type"] if outputs else "chat"


def _infer_satisfaction(user_message: str, assistant_content: str) -> str:
    text = f"{user_message}\n{assistant_content}"
    if any(word in text for word in ("不满意", "不行", "重做", "不要", "不喜欢")):
        return "negative"
    if any(word in text for word in ("满意", "很好", "不错", "可以", "确认", "就这个")):
        return "positive"
    return "unknown"


def _extract_outputs(tool_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for result in tool_results:
        content = result.get("content")
        parsed: Any = content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = content
        candidates = _walk_for_paths(parsed)
        outputs.extend(candidates)
    return outputs


def _walk_for_paths(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and "/storage/" in item:
                found.append({"type": _asset_type_from_path(item, key_lower), "url": item})
            else:
                found.extend(_walk_for_paths(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_for_paths(item))
    elif isinstance(value, str) and "/storage/" in value:
        for path in re.findall(r"/storage/[^\s\"'，,)}\]]+", value):
            found.append({"type": _asset_type_from_path(path, ""), "url": path})
    deduped = []
    seen = set()
    for item in found:
        if item["url"] not in seen:
            seen.add(item["url"])
            deduped.append(item)
    return deduped


def _asset_type_from_path(path: str, key: str) -> str:
    lowered = f"{key} {path}".lower()
    if "/images/" in lowered or lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    if "/videos/" in lowered or lowered.endswith((".mp4", ".mov", ".webm")):
        return "video"
    if "/audios/" in lowered or lowered.endswith((".mp3", ".wav", ".m4a")):
        return "audio"
    if "/models/" in lowered or lowered.endswith((".obj", ".glb", ".gltf")):
        return "3d"
    return "asset"


def _compact_digest(text: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.65)].rstrip()
    tail = text[-int(max_chars * 0.25):].lstrip()
    return f"{head}\n... [会话摘要已压缩] ...\n{tail}"










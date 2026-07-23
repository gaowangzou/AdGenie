"""
聊天路由 - 处理对话请求
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from app.services.agent_service import process_chat_stream
from app.services.history_service import history_service
from app.services.connection_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[Dict[str, Any]]] = []
    session_id: Optional[str] = None
    canvas_id: Optional[str] = None


@router.get("/canvases")
async def get_canvases():
    """获取所有画布历史"""
    return history_service.get_canvases()


@router.post("/canvases")
async def save_canvas(request: Request):
    """保存或更新画布（项目）

    注意：前端会携带 Excalidraw 的 data(elements/appState/files)，
    用 Pydantic 模型解析容易因 extra 字段处理/热重载不同步而丢字段。
    这里直接存原始 JSON，避免 data 被过滤导致刷新后画布空白。
    """
    payload = await request.json()
    return history_service.save_canvas(payload)


@router.delete("/canvases/{canvas_id}")
async def delete_canvas(canvas_id: str):
    """删除画布"""
    history_service.delete_canvas(canvas_id)
    return {"success": True}


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片到 storage/images 目录
    
    Returns:
        上传后的图片URL（相对路径，如 /storage/images/xxx.jpg）
    """
    try:
        # 确保上传目录存在
        BASE_DIR = Path(__file__).parent.parent.parent
        IMAGES_DIR = BASE_DIR / "storage" / "images"
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 验证文件类型
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # 获取文件扩展名
        original_filename = file.filename or "image"
        ext = os.path.splitext(original_filename)[1] or ".jpg"
        if not ext.startswith("."):
            ext = ".jpg"
        
        filename = f"upload_{timestamp}_{unique_id}{ext}"
        file_path = IMAGES_DIR / filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 返回相对路径
        image_url = f"/storage/images/{filename}"
        return {"url": image_url, "filename": filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    上传音频到 storage/audios 目录
    支持的格式：mp3, wav, m4a, aac, ogg, flac, wma
    
    Returns:
        上传后的音频URL（相对路径，如 /storage/audios/xxx.mp3 或 /storage/audios/xxx.wav）
    """
    try:
        # 确保上传目录存在
        BASE_DIR = Path(__file__).parent.parent.parent
        AUDIOS_DIR = BASE_DIR / "storage" / "audios"
        AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 验证文件类型
        content_type = file.content_type or ""
        # 支持的音频 MIME 类型
        allowed_audio_types = [
            "audio/",  # 通用音频类型（audio/mpeg, audio/wav, audio/mp4 等）
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/mpeg",  # mp3
            "audio/mp4",  # m4a
            "audio/aac",
            "audio/ogg",
            "audio/flac",
            "application/octet-stream"  # 有些浏览器可能不识别音频类型
        ]
        is_audio = any(content_type.startswith(t) for t in allowed_audio_types)
        
        # 也检查文件扩展名
        original_filename = file.filename or "audio"
        ext = os.path.splitext(original_filename)[1].lower()
        allowed_extensions = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"]
        
        if not is_audio and ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"只支持音频文件，支持的格式：{', '.join(allowed_extensions)}"
            )
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # 获取文件扩展名，如果没有则默认使用 .mp3
        if not ext or ext not in allowed_extensions:
            ext = ".mp3"
        
        filename = f"upload_{timestamp}_{unique_id}{ext}"
        file_path = AUDIOS_DIR / filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 返回相对路径
        audio_url = f"/storage/audios/{filename}"
        return {"url": audio_url, "filename": filename}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频到 storage/videos 目录
    支持的格式：mp4, mov, avi, mkv, webm

    Returns:
        上传后的视频URL（相对路径，如 /storage/videos/xxx.mp4）
    """
    try:
        BASE_DIR = Path(__file__).parent.parent.parent
        VIDEOS_DIR = BASE_DIR / "storage" / "videos"
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        content_type = file.content_type or ""
        original_filename = file.filename or "video"
        ext = os.path.splitext(original_filename)[1].lower()
        allowed_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

        is_video = content_type.startswith("video/") or content_type == "application/octet-stream"
        if not is_video and ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"只支持视频文件，支持的格式：{', '.join(allowed_extensions)}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        if not ext or ext not in allowed_extensions:
            ext = ".mp4"

        filename = f"upload_{timestamp}_{unique_id}{ext}"
        file_path = VIDEOS_DIR / filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        video_url = f"/storage/videos/{filename}"
        return {"url": video_url, "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求，返回流式响应
    支持OpenAI格式的流式输出
    若请求包含 canvas_id，同时将每个 SSE 事件广播给订阅该 canvas 的 WebSocket 客户端
    """
    try:
        # 构建消息历史
        messages = request.messages.copy() if request.messages else []
        messages.append({
            "role": "user",
            "content": request.message
        })

        canvas_id = request.canvas_id

        async def stream_and_save():
            assistant_content = ""
            # 收集工具调用结果（图片/视频等媒体 URL）
            tool_results: list[dict] = []

            # 广播用户消息给前端（await 直接调用，不用 create_task）
            user_event = json.dumps({"type": "user_message", "content": request.message}, ensure_ascii=False)
            if canvas_id:
                await manager.broadcast(canvas_id, user_event)
            else:
                await manager.broadcast_all(user_event)

            async for chunk in process_chat_stream(messages, request.session_id):
                yield chunk
                if chunk.startswith("data: "):
                    data_str = chunk[len("data: "):].strip()
                    if data_str and data_str != "[DONE]":
                        # 广播给前端
                        if canvas_id:
                            await manager.broadcast(canvas_id, data_str)
                        else:
                            await manager.broadcast_all(data_str)
                        # 收集 assistant 文本内容与工具调用结果
                        try:
                            ev = json.loads(data_str)
                            ev_type = ev.get("type")
                            if ev_type == "delta" and ev.get("content"):
                                assistant_content += ev["content"]
                            elif ev_type == "tool_result":
                                # 保留工具调用结果（含图片/视频 URL 等）
                                tool_results.append({
                                    "tool_call_id": ev.get("tool_call_id"),
                                    "content": ev.get("content"),
                                })
                        except Exception:
                            pass
                    elif data_str == "[DONE]":
                        done_event = json.dumps({"type": "done"}, ensure_ascii=False)
                        if canvas_id:
                            await manager.broadcast(canvas_id, done_event)
                        else:
                            await manager.broadcast_all(done_event)

            # 流结束后保存历史
            try:
                import time
                ts = int(time.time() * 1000)

                # 构建本轮新增的消息（用户 + 助手）
                new_messages = [
                    {"role": "user", "content": request.message},
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        # 若有工具调用结果（图片等），附加到 assistant 消息
                        **({"tool_results": tool_results} if tool_results else {}),
                    },
                ]

                if canvas_id:
                    # 已有项目：加载原有 canvas，追加本轮消息，不新建项目
                    existing_canvases = history_service.get_canvases()
                    existing = next((c for c in existing_canvases if c.get("id") == canvas_id), None)
                    if existing:
                        existing_messages = existing.get("messages", [])
                        existing["messages"] = existing_messages + new_messages
                        await asyncio.to_thread(history_service.save_canvas, existing)
                    else:
                        # canvas_id 在历史中找不到（可能已删除），退化为新建
                        canvas_to_save = {
                            "id": canvas_id,
                            "name": request.message[:30],
                            "createdAt": ts,
                            "messages": new_messages,
                            "data": {"elements": [], "appState": {}, "files": {}},
                        }
                        await asyncio.to_thread(history_service.save_canvas, canvas_to_save)
                else:
                    # 没有 canvas_id：新建项目
                    new_canvas = {
                        "id": f"canvas-{ts}",
                        "name": request.message[:30],
                        "createdAt": ts,
                        "messages": new_messages,
                        "data": {"elements": [], "appState": {}, "files": {}},
                    }
                    await asyncio.to_thread(history_service.save_canvas, new_canvas)
            except Exception as e:
                logger.warning(f"保存对话历史失败: {e}")

        # 返回流式响应 - 确保立即发送，不缓冲
        return StreamingResponse(
            stream_and_save(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


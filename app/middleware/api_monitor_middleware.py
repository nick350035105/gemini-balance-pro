"""
API 监控中间件 - 记录完整的请求和响应内容到数据库
"""
import asyncio
import json
import re
import time
from datetime import datetime
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from app.log.logger import get_request_logger

logger = get_request_logger()

# 最大存储长度
MAX_BODY_LENGTH = 50000


def truncate_body(data: str, max_length: int = MAX_BODY_LENGTH) -> str:
    """截断过长的内容"""
    if len(data) > max_length:
        return data[:max_length] + f"\n... (截断，共 {len(data)} 字符)"
    return data


def redact_api_key(key: Optional[str]) -> Optional[str]:
    """脱敏 API 密钥"""
    if not key:
        return None
    if len(key) <= 12:
        return key[:3] + "..." + key[-3:] if len(key) > 6 else "***"
    return key[:6] + "..." + key[-6:]


def extract_model_from_body(body_str: str) -> Optional[str]:
    """从请求体中提取模型名称"""
    try:
        data = json.loads(body_str)
        return data.get("model")
    except:
        return None


def extract_content_preview(body_str: str, max_length: int = 12) -> Optional[str]:
    """从请求体中提取内容预览（前12个字）"""
    try:
        data = json.loads(body_str)
        # OpenAI 格式: messages[0].content
        messages = data.get("messages", [])
        if messages:
            for msg in messages:
                content = msg.get("content")
                if content:
                    if isinstance(content, str):
                        # 去除换行和多余空格
                        content = " ".join(content.split())
                        return content[:max_length] if len(content) > max_length else content
                    elif isinstance(content, list):
                        # 多模态内容
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                text = " ".join(text.split())
                                return text[:max_length] if len(text) > max_length else text
        # Gemini 格式: contents[0].parts[0].text
        contents = data.get("contents", [])
        if contents:
            for c in contents:
                parts = c.get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        text = part.get("text", "")
                        text = " ".join(text.split())
                        return text[:max_length] if len(text) > max_length else text
        # prompt 字段
        prompt = data.get("prompt")
        if prompt:
            prompt = " ".join(prompt.split())
            return prompt[:max_length] if len(prompt) > max_length else prompt
    except:
        pass
    return None


def extract_tokens_from_response(response_str: str) -> tuple:
    """从响应体中提取 token 统计信息"""
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    try:
        data = json.loads(response_str)
        # OpenAI 格式
        usage = data.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        # Gemini 格式: usageMetadata
        usage_metadata = data.get("usageMetadata", {})
        if usage_metadata:
            prompt_tokens = usage_metadata.get("promptTokenCount")
            completion_tokens = usage_metadata.get("candidatesTokenCount")
            total_tokens = usage_metadata.get("totalTokenCount")
    except:
        pass
    return prompt_tokens, completion_tokens, total_tokens


def extract_model_from_path(path: str) -> Optional[str]:
    """从路径中提取模型名称"""
    # 匹配 /models/xxx:generateContent 等模式
    match = re.search(r'/models/([^/:]+)', path)
    if match:
        return match.group(1)
    return None


class APIMonitorMiddleware(BaseHTTPMiddleware):
    """API 监控中间件，记录请求和响应的完整内容到数据库"""

    # 需要监控的路径前缀
    MONITOR_PATHS = ["/v1/", "/gemini/", "/openai/", "/hf/", "/vertex-express/"]

    # 不需要监控的路径
    EXCLUDE_PATHS = ["/static/", "/health", "/favicon.ico"]

    def should_monitor(self, path: str) -> bool:
        """判断是否需要监控该路径"""
        for exclude in self.EXCLUDE_PATHS:
            if path.startswith(exclude):
                return False

        for monitor in self.MONITOR_PATHS:
            if path.startswith(monitor):
                return True

        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.should_monitor(request.url.path):
            return await call_next(request)

        start_time = time.time()
        request_time = datetime.now()

        # ========== 收集请求信息 ==========
        method = request.method
        path = request.url.path

        # 请求头（只记录重要的，脱敏处理）
        important_headers = ['content-type', 'authorization', 'user-agent', 'x-api-key']
        headers_dict = {}
        api_key = None

        for k, v in request.headers.items():
            if k.lower() in important_headers:
                if k.lower() in ['authorization', 'x-api-key']:
                    # 提取并脱敏 API key
                    if k.lower() == 'authorization' and v.startswith('Bearer '):
                        api_key = v[7:]
                    elif k.lower() == 'x-api-key':
                        api_key = v
                    headers_dict[k] = redact_api_key(v) or v
                else:
                    headers_dict[k] = v

        request_headers_str = json.dumps(headers_dict, ensure_ascii=False)

        # 读取请求体
        body = b""
        request_body_str = ""
        model_name = None

        try:
            body = await request.body()
            if body:
                request_body_str = body.decode('utf-8', errors='replace')
                # 从请求体提取模型名称
                model_name = extract_model_from_body(request_body_str)
        except Exception as e:
            request_body_str = f"[Error reading body: {e}]"

        # 如果请求体中没有模型名，尝试从路径提取
        if not model_name:
            model_name = extract_model_from_path(path)

        # 重置请求体以便后续处理
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request._receive = receive

        # ========== 执行请求 ==========
        response = await call_next(request)

        duration_ms = int((time.time() - start_time) * 1000)
        status_code = response.status_code

        # ========== 收集响应信息 ==========
        response_body = b""

        if isinstance(response, StreamingResponse):
            async for chunk in response.body_iterator:
                response_body += chunk
        else:
            async for chunk in response.body_iterator:
                response_body += chunk

        response_body_str = ""
        if response_body:
            response_body_str = response_body.decode('utf-8', errors='replace')

        # 提取 token 信息
        prompt_tokens, completion_tokens, total_tokens = extract_tokens_from_response(response_body_str)

        # 提取内容预览
        content_preview = extract_content_preview(request_body_str)

        # ========== 异步保存到数据库 ==========
        asyncio.create_task(self._save_log(
            method=method,
            path=path,
            request_headers=request_headers_str,
            request_body=truncate_body(request_body_str),
            status_code=status_code,
            response_body=truncate_body(response_body_str),
            latency_ms=duration_ms,
            model_name=model_name,
            api_key=redact_api_key(api_key),
            request_time=request_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            content_preview=content_preview,
        ))

        # 重新创建响应
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )

    async def _save_log(
        self,
        method: str,
        path: str,
        request_headers: str,
        request_body: str,
        status_code: int,
        response_body: str,
        latency_ms: int,
        model_name: Optional[str],
        api_key: Optional[str],
        request_time: datetime,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        content_preview: Optional[str] = None,
    ):
        """异步保存日志到数据库"""
        try:
            from app.service.ai_log import ai_log_service
            await ai_log_service.add_ai_log(
                method=method,
                path=path,
                request_headers=request_headers,
                request_body=request_body,
                status_code=status_code,
                response_body=response_body,
                latency_ms=latency_ms,
                model_name=model_name,
                api_key=api_key,
                request_time=request_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                content_preview=content_preview,
            )
        except Exception as e:
            logger.error(f"Failed to save AI log to database: {e}")

"""
AI 日志服务模块
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, insert, select, desc, asc

from app.database.connection import database
from app.database.models import AILog
from app.log.logger import get_request_logger

logger = get_request_logger()

# AI 日志保留天数
AI_LOG_RETENTION_DAYS = 3


async def add_ai_log(
    method: Optional[str] = None,
    path: Optional[str] = None,
    request_headers: Optional[str] = None,
    request_body: Optional[str] = None,
    status_code: Optional[int] = None,
    response_body: Optional[str] = None,
    latency_ms: Optional[int] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    request_time: Optional[datetime] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    content_preview: Optional[str] = None,
) -> bool:
    """
    添加 AI 日志记录

    Args:
        method: HTTP 方法
        path: 请求路径
        request_headers: 请求头 JSON
        request_body: 请求体
        status_code: 响应状态码
        response_body: 响应体
        latency_ms: 延迟毫秒
        model_name: 模型名称
        api_key: API 密钥 (脱敏)
        request_time: 请求时间
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        total_tokens: 总 token 数
        content_preview: 内容预览

    Returns:
        bool: 是否添加成功
    """
    try:
        query = insert(AILog).values(
            method=method,
            path=path,
            request_headers=request_headers,
            request_body=request_body,
            status_code=status_code,
            response_body=response_body,
            latency_ms=latency_ms,
            model_name=model_name,
            api_key=api_key,
            request_time=request_time or datetime.now(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            content_preview=content_preview,
        )
        await database.execute(query)
        return True
    except Exception as e:
        logger.error(f"Failed to add AI log: {str(e)}")
        return False


async def get_ai_logs(
    limit: int = 20,
    offset: int = 0,
    model_search: Optional[str] = None,
    status_code_search: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sort_by: str = "id",
    sort_order: str = "desc",
) -> List[Dict[str, Any]]:
    """
    获取 AI 日志列表

    Args:
        limit: 限制数量
        offset: 偏移量
        model_search: 模型名称搜索
        status_code_search: 状态码搜索
        start_date: 开始日期
        end_date: 结束日期
        sort_by: 排序字段
        sort_order: 排序顺序

    Returns:
        List[Dict[str, Any]]: AI 日志列表
    """
    try:
        query = select(
            AILog.id,
            AILog.model_name,
            AILog.request_time,
            AILog.content_preview,
            AILog.prompt_tokens,
            AILog.completion_tokens,
            AILog.total_tokens,
            AILog.status_code,
            AILog.latency_ms,
        )

        # 应用过滤条件
        if model_search:
            query = query.where(AILog.model_name.ilike(f"%{model_search}%"))
        if status_code_search:
            try:
                code = int(status_code_search)
                query = query.where(AILog.status_code == code)
            except ValueError:
                pass
        if start_date:
            query = query.where(AILog.request_time >= start_date)
        if end_date:
            query = query.where(AILog.request_time <= end_date)

        # 应用排序
        sort_column = getattr(AILog, sort_by, AILog.id)
        if sort_order.lower() == "asc":
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))

        # 应用分页
        query = query.limit(limit).offset(offset)

        result = await database.fetch_all(query)
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Failed to get AI logs: {str(e)}")
        raise


async def get_ai_logs_count(
    model_search: Optional[str] = None,
    status_code_search: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """
    获取 AI 日志总数

    Returns:
        int: 日志总数
    """
    try:
        query = select(func.count(AILog.id))

        if model_search:
            query = query.where(AILog.model_name.ilike(f"%{model_search}%"))
        if status_code_search:
            try:
                code = int(status_code_search)
                query = query.where(AILog.status_code == code)
            except ValueError:
                pass
        if start_date:
            query = query.where(AILog.request_time >= start_date)
        if end_date:
            query = query.where(AILog.request_time <= end_date)

        result = await database.fetch_val(query)
        return result or 0
    except Exception as e:
        logger.error(f"Failed to get AI logs count: {str(e)}")
        raise


async def get_ai_log_details(log_id: int) -> Optional[Dict[str, Any]]:
    """
    获取单条 AI 日志详情

    Args:
        log_id: 日志 ID

    Returns:
        Optional[Dict[str, Any]]: 日志详情
    """
    try:
        query = select(AILog).where(AILog.id == log_id)
        result = await database.fetch_one(query)
        return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get AI log details for ID {log_id}: {str(e)}")
        raise


async def delete_ai_log_by_id(log_id: int) -> bool:
    """
    删除单条 AI 日志

    Args:
        log_id: 日志 ID

    Returns:
        bool: 是否删除成功
    """
    try:
        query = delete(AILog).where(AILog.id == log_id)
        result = await database.execute(query)
        return True
    except Exception as e:
        logger.error(f"Failed to delete AI log ID {log_id}: {str(e)}")
        return False


async def delete_ai_logs_by_ids(log_ids: List[int]) -> int:
    """
    批量删除 AI 日志

    Args:
        log_ids: 日志 ID 列表

    Returns:
        int: 删除数量
    """
    try:
        query = delete(AILog).where(AILog.id.in_(log_ids))
        await database.execute(query)
        return len(log_ids)
    except Exception as e:
        logger.error(f"Failed to delete AI logs by IDs: {str(e)}")
        raise


async def delete_all_ai_logs() -> int:
    """
    删除所有 AI 日志

    Returns:
        int: 删除数量
    """
    try:
        # 先获取总数
        count_query = select(func.count(AILog.id))
        count = await database.fetch_val(count_query)

        # 删除所有
        query = delete(AILog)
        await database.execute(query)

        logger.info(f"Deleted all AI logs, count: {count}")
        return count or 0
    except Exception as e:
        logger.error(f"Failed to delete all AI logs: {str(e)}")
        raise


async def delete_old_ai_logs(days: int = AI_LOG_RETENTION_DAYS) -> int:
    """
    删除超过指定天数的 AI 日志

    Args:
        days: 保留天数，默认 3 天

    Returns:
        int: 删除数量
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)

        # 先统计要删除的数量
        count_query = select(func.count(AILog.id)).where(
            AILog.request_time < cutoff_date
        )
        count = await database.fetch_val(count_query)

        if count == 0:
            logger.info(f"No AI logs older than {days} days to delete.")
            return 0

        # 执行删除
        query = delete(AILog).where(AILog.request_time < cutoff_date)
        await database.execute(query)

        logger.info(f"Deleted {count} AI logs older than {days} days.")
        return count or 0
    except Exception as e:
        logger.error(f"Failed to delete old AI logs: {str(e)}")
        raise

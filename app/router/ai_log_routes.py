"""
AI 日志路由模块
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel

from app.core.security import verify_auth_token
from app.log.logger import get_log_routes_logger
from app.service.ai_log import ai_log_service

router = APIRouter(prefix="/api/ai-logs", tags=["ai-logs"])

logger = get_log_routes_logger()


class AILogListItem(BaseModel):
    id: int
    model_name: Optional[str] = None
    request_time: Optional[datetime] = None
    content_preview: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None


class AILogListResponse(BaseModel):
    logs: List[AILogListItem]
    total: int


class AILogDetailResponse(BaseModel):
    id: int
    request_time: Optional[datetime] = None
    method: Optional[str] = None
    path: Optional[str] = None
    request_headers: Optional[str] = None
    request_body: Optional[str] = None
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    latency_ms: Optional[int] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    content_preview: Optional[str] = None


@router.get("", response_model=AILogListResponse)
async def get_ai_logs_api(
    request: Request,
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model_search: Optional[str] = Query(None, description="搜索模型名称"),
    status_code_search: Optional[str] = Query(None, description="搜索状态码"),
    start_date: Optional[datetime] = Query(None, description="开始时间"),
    end_date: Optional[datetime] = Query(None, description="结束时间"),
    sort_by: str = Query("id", description="排序字段"),
    sort_order: str = Query("desc", description="排序顺序"),
):
    """
    获取 AI 日志列表
    """
    auth_token = request.cookies.get("auth_token")
    if not auth_token or not verify_auth_token(auth_token):
        logger.warning("Unauthorized access attempt to AI logs list")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        logs_data = await ai_log_service.get_ai_logs(
            limit=limit,
            offset=offset,
            model_search=model_search,
            status_code_search=status_code_search,
            start_date=start_date,
            end_date=end_date,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        total_count = await ai_log_service.get_ai_logs_count(
            model_search=model_search,
            status_code_search=status_code_search,
            start_date=start_date,
            end_date=end_date,
        )

        validated_logs = [AILogListItem(**log) for log in logs_data]
        return AILogListResponse(logs=validated_logs, total=total_count)
    except Exception as e:
        logger.exception(f"Failed to get AI logs list: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get AI logs list: {str(e)}"
        )


@router.get("/{log_id}/details", response_model=AILogDetailResponse)
async def get_ai_log_detail_api(request: Request, log_id: int = Path(..., ge=1)):
    """
    获取单条 AI 日志详情
    """
    auth_token = request.cookies.get("auth_token")
    if not auth_token or not verify_auth_token(auth_token):
        logger.warning(f"Unauthorized access attempt to AI log details for ID: {log_id}")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        log_details = await ai_log_service.get_ai_log_details(log_id=log_id)
        if not log_details:
            raise HTTPException(status_code=404, detail="AI log not found")

        return AILogDetailResponse(**log_details)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Failed to get AI log details for ID {log_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get AI log details: {str(e)}"
        )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ai_logs_bulk_api(
    request: Request, payload: Dict[str, List[int]] = Body(...)
):
    """
    批量删除 AI 日志
    """
    auth_token = request.cookies.get("auth_token")
    if not auth_token or not verify_auth_token(auth_token):
        logger.warning("Unauthorized access attempt to bulk delete AI logs")
        raise HTTPException(status_code=401, detail="Not authenticated")

    log_ids = payload.get("ids")
    if not log_ids:
        raise HTTPException(status_code=400, detail="No log IDs provided for deletion.")

    try:
        deleted_count = await ai_log_service.delete_ai_logs_by_ids(log_ids)
        logger.info(f"Bulk deleted {deleted_count} AI logs with IDs: {log_ids}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.exception(f"Error bulk deleting AI logs with IDs {log_ids}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during bulk deletion"
        )


@router.delete("/all", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_ai_logs_api(request: Request):
    """
    删除所有 AI 日志
    """
    auth_token = request.cookies.get("auth_token")
    if not auth_token or not verify_auth_token(auth_token):
        logger.warning("Unauthorized access attempt to delete all AI logs")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        deleted_count = await ai_log_service.delete_all_ai_logs()
        logger.info(f"Successfully deleted all AI logs. Count: {deleted_count}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.exception(f"Error deleting all AI logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during deletion of all logs"
        )


@router.delete("/{log_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ai_log_api(request: Request, log_id: int = Path(..., ge=1)):
    """
    删除单条 AI 日志
    """
    auth_token = request.cookies.get("auth_token")
    if not auth_token or not verify_auth_token(auth_token):
        logger.warning(f"Unauthorized access attempt to delete AI log ID: {log_id}")
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        success = await ai_log_service.delete_ai_log_by_id(log_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"AI log with ID {log_id} not found"
            )
        logger.info(f"Successfully deleted AI log with ID: {log_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error deleting AI log with ID {log_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during deletion"
        )

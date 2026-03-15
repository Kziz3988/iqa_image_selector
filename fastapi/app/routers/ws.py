from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.progress import progress_manager
from app.config import WS_ROUTE

router = APIRouter()
@router.websocket(f"{WS_ROUTE}/{{task_id}}")

async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await progress_manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_manager.disconnect(task_id)
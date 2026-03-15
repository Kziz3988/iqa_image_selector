from typing import Dict
from fastapi import WebSocket

class ProgressManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[task_id] = websocket

    def disconnect(self, task_id: str):
        if task_id in self.connections:
            del self.connections[task_id]

    async def send(self, task_id: str, message: dict):
        ws = self.connections.get(task_id)
        if ws:
            await ws.send_json(message)

progress_manager = ProgressManager()
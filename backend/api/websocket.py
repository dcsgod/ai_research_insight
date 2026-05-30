"""
AI Research Intelligence Platform — WebSocket Live Feed
Real-time streaming of trending updates, new papers, and score changes.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """
    Manages active WebSocket connections.
    Supports broadcast, targeted messages, and graceful disconnection.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WS client connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WS client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a message to all connected clients."""
        if not self.active_connections:
            return
        payload = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        for ws in list(self.active_connections):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        async with self._lock:
            for ws in dead:
                self.active_connections.discard(ws)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# Singleton connection manager
manager = ConnectionManager()


def _build_trend_update_message() -> Dict[str, Any]:
    """Build a mock trend update message (replace with real DB query in prod)."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "type": "trend_update",
        "timestamp": now,
        "data": {
            "top_topics": [
                {"name": "GraphRAG", "score": 0.94, "delta": +0.12},
                {"name": "Mixture of Experts", "score": 0.87, "delta": +0.08},
                {"name": "Multimodal LLMs", "score": 0.81, "delta": +0.05},
                {"name": "RLHF / DPO", "score": 0.76, "delta": +0.03},
                {"name": "Vision Transformers", "score": 0.71, "delta": -0.02},
            ],
            "active_connections": manager.connection_count,
        },
    }


def _build_heartbeat_message() -> Dict[str, Any]:
    return {
        "type": "heartbeat",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "connections": manager.connection_count,
    }


@router.websocket("/ws/live-feed")
async def websocket_live_feed(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI trend updates.

    Message types sent to clients:
    - `heartbeat`: Sent every 10s to keep connection alive
    - `trend_update`: Top trending items, sent every 30s
    - `new_paper`: When a new high-score paper is ingested
    - `score_update`: When trend scores are recomputed
    - `connected`: Sent once on connection with initial state
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        await manager.send_personal(websocket, {
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to AI Research Intelligence Platform live feed",
            "version": "1.0.0",
        })

        # Start background task that sends periodic updates to this client
        heartbeat_task = asyncio.create_task(
            _send_periodic_updates(websocket)
        )

        # Listen for client messages (e.g., filter preferences)
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
                msg = json.loads(raw)
                await _handle_client_message(websocket, msg)
            except asyncio.TimeoutError:
                # Client inactive — send heartbeat and continue
                await manager.send_personal(websocket, _build_heartbeat_message())
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        await manager.disconnect(websocket)


async def _send_periodic_updates(websocket: WebSocket) -> None:
    """Background task: send trend updates every 30s and heartbeats every 10s."""
    tick = 0
    try:
        while True:
            await asyncio.sleep(10)
            tick += 1

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if tick % 3 == 0:
                # Every 30s — send full trend update
                await manager.send_personal(websocket, _build_trend_update_message())
            else:
                # Every 10s — heartbeat
                await manager.send_personal(websocket, _build_heartbeat_message())

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Periodic update error: {e}")


async def _handle_client_message(websocket: WebSocket, msg: Dict[str, Any]) -> None:
    """Handle incoming messages from client (e.g., subscribe to topics)."""
    msg_type = msg.get("type", "")

    if msg_type == "ping":
        await manager.send_personal(websocket, {
            "type": "pong",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    elif msg_type == "request_update":
        await manager.send_personal(websocket, _build_trend_update_message())
    elif msg_type == "subscribe":
        topics = msg.get("topics", [])
        await manager.send_personal(websocket, {
            "type": "subscribed",
            "topics": topics,
            "message": f"Subscribed to: {', '.join(topics)}",
        })
    else:
        await manager.send_personal(websocket, {
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
        })


async def broadcast_new_paper(paper_data: Dict[str, Any]) -> None:
    """Called externally when a new high-scoring paper is ingested."""
    await manager.broadcast({
        "type": "new_paper",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": paper_data,
    })


async def broadcast_score_update(entity_id: str, new_score: float, entity_type: str) -> None:
    """Called externally when trend scores are recomputed."""
    await manager.broadcast({
        "type": "score_update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "new_score": new_score,
        },
    })

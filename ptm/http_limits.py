from typing import Any

from starlette.responses import JSONResponse

from ptm.errors import PTMError


class BodyLimitMiddleware:
    """Bound mutation bodies even when Content-Length is absent (chunked requests)."""
    def __init__(self, app: Any, max_bytes: int = 8*1024*1024) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        data = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data.extend(message.get("body", b""))
            if len(data) > self.max_bytes:
                response = JSONResponse(PTMError("PTM_IMPORT_LIMIT", "Request exceeds 8 MiB").as_dict(), 413)
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Any:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(data), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

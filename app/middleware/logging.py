import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        start_time = time.time()

        logger.info(
            f"Incoming Request -> {request.method} {request.url.path}"
        )

        response = await call_next(request)

        process_time = round(time.time() - start_time, 3)

        logger.info(
            f"Completed -> {response.status_code} | {process_time}s"
        )

        response.headers["X-Process-Time"] = str(process_time)

        return response
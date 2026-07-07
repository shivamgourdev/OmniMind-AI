from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import logger


def register_exception_handlers(app: FastAPI):

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Validation Error",
                # str(exc) instead of exc.errors(): some pydantic v2 error
                # contexts embed non-JSON-serializable objects, which broke
                # this handler previously. Keeping the string form here.
                "errors": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ):

        # Always log the full error server-side.
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")

        # Only leak the raw exception message to the client in DEBUG mode.
        # In production, exposing exception internals to callers is an
        # information-disclosure risk (stack traces, file paths, library
        # versions, sometimes fragments of query data).
        detail = str(exc) if settings.DEBUG else "An unexpected error occurred."

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal Server Error",
                "error": detail,
            },
        )

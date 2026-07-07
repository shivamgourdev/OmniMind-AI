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

                "errors": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ):


        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")

    
        detail = str(exc) if settings.DEBUG else "An unexpected error occurred."

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal Server Error",
                "error": detail,
            },
        )

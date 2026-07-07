import logging
import os

from app.core.config import settings

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(settings.APP_NAME)
logger.setLevel(getattr(logging, settings.LOG_LEVEL))

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = logging.FileHandler(
    "logs/omnimind.log",
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.propagate = False

logger.info("Logger Initialized Successfully.")
from __future__ import annotations

import logging
import sys

from loguru import logger


def _ensure_class_name(record: dict) -> None:
    record["extra"].setdefault("class_name", "-")


app_logger = logger.patch(_ensure_class_name)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = app_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        app_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str) -> None:
    app_logger.remove()
    app_logger.add(
        sys.stdout,
        level=level,
        colorize=False,
        enqueue=False,
        backtrace=True,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{file.name}:{line} | {name}.{function} | class={extra[class_name]} | {message}"
        ),
    )

    intercept_handler = InterceptHandler()

    logging.root.handlers = [intercept_handler]
    logging.root.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        named_logger = logging.getLogger(logger_name)
        named_logger.handlers = [intercept_handler]
        named_logger.propagate = False

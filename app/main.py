from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.compare import router as compare_router
from app.api.routes.health import router as health_router
from app.core.config import AppSettings, load_settings
from app.core.logging import app_logger, setup_logging


def _log_startup_settings(settings: AppSettings) -> None:
    app_logger.info(
        "Startup config: API_DEBUG={} ({}), IMAGE_BASE_DIR='{}' ({}), LOG_LEVEL={} ({})"
        " | CONFIG_TOML={} | AI_CORNER_CONFIG_TOML={}",
        settings.api_debug,
        settings.sources["API_DEBUG"],
        settings.image_base_dir,
        settings.sources["IMAGE_BASE_DIR"],
        settings.log_level,
        settings.sources["LOG_LEVEL"],
        settings.config_toml_path,
        settings.sources["CONFIG_TOML"],
    )


def create_app() -> FastAPI:
    settings = load_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _log_startup_settings(settings)
        yield

    app = FastAPI(title="Perceptual Metrics Service", lifespan=lifespan)

    app.include_router(health_router)
    app.include_router(compare_router)

    debug = settings.api_debug

    if debug:

        @app.exception_handler(Exception)
        async def unhandled_exception_handler(request: Request, exc: Exception):
            app_logger.exception("Unhandled exception on {} {}", request.method, request.url.path)

            return JSONResponse(
                status_code=500,
                content={
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "path": request.url.path,
                    "method": request.method,
                },
            )

    return app


app = create_app()

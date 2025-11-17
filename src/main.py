"""Main application entry point"""

import uvicorn
from src.config import settings
from src.logging_config import setup_logging, get_logger


def initialize_app() -> None:
    """Initialize the MHRAS application"""
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(
        "application_starting",
        environment=settings.environment,
        debug=settings.debug,
    )
    
    # Log configuration summary (without sensitive data)
    logger.info(
        "configuration_loaded",
        database_host=settings.database.host,
        database_port=settings.database.port,
        api_host=settings.api.host,
        api_port=settings.api.port,
        log_level=settings.logging.level,
        log_format=settings.logging.format,
    )
    
    logger.info("application_initialized")


def run_api_server():
    """Run the FastAPI server"""
    initialize_app()
    
    logger = get_logger(__name__)
    logger.info(
        "starting_api_server",
        host=settings.api.host,
        port=settings.api.port
    )
    
    # Run the API server
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.logging.level.lower()
    )


if __name__ == "__main__":
    run_api_server()

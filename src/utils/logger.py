from datetime import datetime
import logging
import logging.config
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

logger = logging.getLogger("project_logger")

def configure_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG if VERBOSE else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def configure_lighrag_logger():
    """Configure logging for the application, including lightrag + custom insertion logs."""

    # Reset existing handlers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag", "InsertionLogger"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers.clear()
        logger_instance.filters.clear()

    # === Log file setup ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.getenv("LOG_DIR", "./logs/lightrag")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"lightrag_{timestamp}.log")

    print(f"\n📂 Logging to: {log_file_path}\n")

    # === File rotation config ===
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # 5 backups

    # === Configure logging ===
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(levelname)s: %(message)s",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "InsertionLogger": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console", "file"],
        }
    })

    return logging.getLogger("InsertionLogger")
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("nl_bash_api")


def log_request(method: str, url: str, status_code: int, latency_ms: float):
    logger.info(f"{method} {url} -> {status_code} [{latency_ms}ms]")

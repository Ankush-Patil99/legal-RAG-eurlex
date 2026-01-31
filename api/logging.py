import logging
import uuid
import time

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "request_id=%(request_id)s | "
    "%(message)s"
)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT
    )

    logger = logging.getLogger()
    logger.addFilter(RequestIdFilter())

def get_request_id():
    return str(uuid.uuid4())

def now_ms():
    return time.perf_counter() * 1000

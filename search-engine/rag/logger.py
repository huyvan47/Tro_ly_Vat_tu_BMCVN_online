# logger.py
import logging
import uuid

def get_logger(name="rag"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] [%(trace_id)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def new_trace_id():
    return str(uuid.uuid4())[:8]

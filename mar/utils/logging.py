import logging
import sys
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

def get_logger(name="MAR"):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        handler = TqdmLoggingHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.propagate = False
    return log
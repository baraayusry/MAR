import logging, sys

def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
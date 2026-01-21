import sys
import logging


class CustomLogger:
    def __init__(self, name=__name__):
        self.name = name
        self.logger = self._setup()

    def _setup(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[%(levelname)s | %(name)s]\n%(message)s")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_logger(self):
        return self.logger

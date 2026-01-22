import sys
import logging

from pathlib import Path
from logging.config import dictConfig

from omegaconf import OmegaConf


class HAMLogger:
    def __init__(self, name=__name__):
        self.name = name

    def configure_from_file(self, filepath: Path | str) -> None:
        log = logging.getLogger(self.name)
        try:
            cfg = OmegaConf.load(filepath)
            config = OmegaConf.to_object(cfg=cfg)
            dictConfig(config=config)

        except FileNotFoundError:
            log.setLevel(level=logging.INFO)
            handler = logging.StreamHandler(stream=sys.stdout)
            fmt = "%(lineno)s: logging_config_file_path_not_found"
            handler.setFormatter(logging.Formatter(fmt=fmt))
            log.addHandler(handler)

        return log

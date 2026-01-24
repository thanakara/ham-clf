import json

from pathlib import Path

from omegaconf import OmegaConf

from ham_clf.utils.loggers import HAMLogger

logconf_path = Path("conf") / "logconf.yaml"
config_path = logconf_path.with_name("config.yaml")
logger = HAMLogger()
log = logger.configure_from_file(logconf_path)


def main():
    cfg = OmegaConf.load(config_path)
    OmegaConf.register_new_resolver("lower_lr", resolver=lambda lr: 0.1 * lr)
    config_dict = OmegaConf.to_object(cfg=cfg)
    log.info(json.dumps(config_dict, indent=3))


if __name__ == "__main__":
    main()

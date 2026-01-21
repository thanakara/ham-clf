import datetime

from ham_clf.utils.loggers import CustomLogger

logger = CustomLogger(name="application")
log = logger.get_logger()


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    log.info(f"ham_mobilenetv2_ft_{now}")

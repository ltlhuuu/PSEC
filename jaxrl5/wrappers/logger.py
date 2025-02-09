import logging


def get_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: - %(message)s"
    )

    # use FileHandler to file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

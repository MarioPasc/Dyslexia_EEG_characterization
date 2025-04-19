import logging

# Set up module-level logger
logger = logging.getLogger("pyddeeg.classification")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = ["logger"]
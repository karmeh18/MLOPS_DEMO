import logging
import os
from datetime import datetime

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_dir, log_file)

# Get the root logger
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

# Prevent adding handlers multiple times
if not rootLogger.handlers:
    logFormatter = logging.Formatter("[%(asctime)s] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s")

    # File Handler
    fileHandler = logging.FileHandler(log_file_path)
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # Console Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

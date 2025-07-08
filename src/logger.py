import logging
import os
from datetime import datetime

# Generate log filename with timestamp
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
log_filename = f"{timestamp}.log"

# Define directory and full path for log storage
log_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(log_directory, exist_ok=True)

log_file_path = os.path.join(log_directory, log_filename)

# Configure logging settings
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] [Line: %(lineno)d] [%(name)s] - %(levelname)s - %(message)s",
    level=logging.INFO
)


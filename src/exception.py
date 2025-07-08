import sys
from src.logger import logging


def format_error_message(error, error_info: sys):
    """
    Constructs a detailed error message including script name, line number, and exception details.
    """
    _, _, traceback_obj = error_info.exc_info()
    file_name = traceback_obj.tb_frame.f_code.co_filename
    line_number = traceback_obj.tb_lineno
    message = f"Exception occurred in file [{file_name}], line [{line_number}]: {str(error)}"
    return message


class CustomException(Exception):
    """
    Custom exception class to enrich Python exceptions with file name and line number context.
    """
    def __init__(self, message, error_info: sys):
        super().__init__(message)
        self.full_message = format_error_message(message, error_info)

    def __str__(self):
        return self.full_message

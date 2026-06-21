"""Base exception classes for MHRAS"""


class MHRASException(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MHRASException):
    pass


class ConsentError(MHRASException):
    pass


class InferenceError(MHRASException):
    pass

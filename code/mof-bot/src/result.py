from enum import Enum, auto

class ResultType(Enum):
    TYPE_FOOLS_CHOICE = auto()
    TYPE_ERROR = auto()
    # Add more result types as needed

class Result:
    def __init__(self, result_type, content):
        """
        Standardized result object to store the type and content of a result.

        Args:
            result_type (ResultType): An instance of ResultType (e.g., ResultType.TYPE_FOOLS_CHOICE).
            content (any): The main content of the result, such as selected posts or an error message.
        """
        self.result_type = result_type
        self.content = content

    def __repr__(self):
        return f"Result(type={self.result_type.name}, content={self.content})"

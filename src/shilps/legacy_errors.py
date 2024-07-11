
class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class ExperimentError(Exception):
    """Exception raised for errors in the experiment results such as infeasibility.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class ProceduralError(Exception):
    """Exception raised for errors in the order of execution of any procedure.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class NotSupportedYet(Exception):
    """Exception raised when the input defines a procedure with a feature that is not implemented
    yet. By design, the input structure considers it so it is not an InputError.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
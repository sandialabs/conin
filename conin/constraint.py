from conin.exceptions import InvalidInputError


# One could also create an inheirted class for additional functionality
# TODO think about partial_func semantics


class Constraint:
    def __init__(
        self,
        *,
        func=None,
        name=None,
        partial_func=None,
        same_partial_as_func=None,
    ):
        """
        Initialize a Constraint object.

        Parameters:
            func (callable, optional): The constraint function to be applied.
            name (str, optional): The name of the constraint. If not provided, it will default to the function's name.
            partial_func (callable, optional): This is a function which returns false on a partial sequence of
                hidden states only if there is no completion of the sequence which satisfies the constraints. E.g. if
                you go over budget halfway through, you can never recover from that. This has, as inputs T and seq instead
                of just seq like for func. This is useful for things like has minimum_number_of_occurences
            same_partial_as_func (bool, optional): If this is true, partial_func is set to func
        """
        self.func = func

        if same_partial_as_func is True:
            self.partial_func = lambda T, seq: func(seq)
        elif partial_func is not None:
            self.partial_func = partial_func
        else:
            self.partial_func = lambda T, seq: True

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name

        elif func is not None:
            self.name = func.__name__

        else:
            self.name = "Unnamed constraint."  # Could also be none

    def __call__(self, seq):
        """
        Apply the constraint function to a given sequence.

        Parameters:
            seq (iterable): The sequence to which the constraint function will be applied.

        Returns:
            The result of applying the constraint function to the sequence.

        Raises:
            InvalidInputError: If the constraint function is not defined.
        """
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )
        return self.func(seq)

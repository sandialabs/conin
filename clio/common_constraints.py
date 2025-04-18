from clio.constraint import Constraint

# TODO Python deals with lambda functions slightly differently than how I thought. I need to double check we are doing things correctly here.


def all_diff(seq):
    """
    Constraint function to check if all elements in the sequence are different

    Parameters:
        seq (iterable): The sequence to be checked.

    Returns:
        bool: True if all elements are unique, False otherwise.
    """
    return len(seq) == len(set(seq))


all_diff_constraint = Constraint(func=all_diff, same_partial_as_func=True)


def always_appears_before(seq, val1, val2):
    """
    Constraint function which means that any instances of val1 appear before any instances of val2

    Parameters:
        seq (iterable): The sequence to be checked.
        val1: val1 appears before val2
        val2: val2 appears after val1

    Returns:
        bool: True iff satisfied
    """
    for index, x1 in enumerate(seq):
        if x1 == val2:
            for index2 in range(index + 1, len(seq)):
                if seq[index2] == val1:
                    return False
            return True
    return True


def always_appears_before_constraint(val1, val2):
    return Constraint(
        func=lambda seq: always_appears_before(seq, val1, val2),
        same_partial_as_func=True,
    )


def appears_at_least_once_before(seq, val1, val2):
    """
    Requires at least one instance of val1 to appear before first val2

    Parameters:
        seq (iterable): The sequence to be checked.
        val1: val1 appears before val2
        val2: val2 appears after val1

    Returns:
        bool: True iff satisfied
    """
    for index1, x1 in enumerate(seq):
        if x1 == val2:
            for index2 in range(0, index1):
                if seq[index2] == val1:
                    return True
            return False
    return True


def appears_at_least_once_before_constraint(val1, val2):
    return Constraint(
        func=lambda seq: appears_at_least_once_before(seq, val1, val2),
        same_partial_as_func=True,
    )


def always_appears_after(seq, val1, val2):
    """
    Constraint function which means that any instances of val1 appear after any instances of val2

    Parameters:
        seq (iterable): The sequence to be checked.
        val1: val1 appears after val2
        val2: val2 appears before val1

    Returns:
        bool: True iff satisfied
    """
    return always_appears_before(seq, val2, val1)


def always_appears_after_constraint(val1, val2):
    return Constraint(
        func=lambda seq: always_appears_after(seq, val1, val2),
        same_partial_as_func=True,
    )


def appears_at_least_once_after(seq, val1, val2):
    """
    Requires at least one instance of val1 to appear after first val2

    Parameters:
        seq (iterable): The sequence to be checked.
        val1: val1 appears after val2
        val2: val2 appears before val1

    Returns:
        bool: True iff satisfied
    """
    for index1, x1 in enumerate(seq):
        if x1 == val2:
            for index2 in range(index1 + 1, len(seq)):
                if seq[index2] == val1:
                    return True
            return False
    return True


def appears_at_least_once_after_constraint(val1, val2):
    # No partial here because val2 could appear at the very last time step
    return Constraint(func=lambda seq: appears_at_least_once_after(seq, val1, val2))


def citation(seq):
    """
    All elements of seq must appear in non-repeating blocks
    E.g.
    True 1112277
    False 11112227722

    Parameters:
        seq (iterable): The sequence to be checked

    Returns:
        bool: True iff satisfied
    """
    for t1 in range(2, len(seq)):
        if seq[t1] != seq[t1 - 1]:
            for t2 in range(t1 - 2):
                if seq[t2] == seq[t1]:
                    return False
    return True


def citation_constraint():
    return Constraint(func=citation, same_partial_as_func=True)


def has_minimum_number_of_occurences(seq, *, val, count):
    """
    Check if seq has val appear at least count times (count times returns true)

    Parameters:
        seq(iterable): The sequence to be checked
        val : Hidden state to be count
        count : min number

    Returns:
        bool: True iff satisfied
    """
    return seq.count(val) >= count


def has_minimum_number_of_occurences_constraint(*, val, count):
    return Constraint(
        func=lambda seq: has_minimum_number_of_occurences(seq, val=val, count=count),
        partial_func=lambda T, seq: seq.count(val) + T - len(seq) >= count,
    )


def has_maximum_number_of_occurences(seq, *, val, count):
    """
    Check if seq has val appear at most count times (count times returns true)

    Parameters:
        seq(iterable): The sequence to be checked
        val : The hidden state to be counter
        count : The max number

    Returns:
        bool: True iff satisfied
    """
    return seq.count(val) <= count


def has_maximum_number_of_occurences_constraint(*, val, count):
    temp_func = lambda seq: has_maximum_number_of_occurences(seq, val=val, count=count)
    return Constraint(func=temp_func, same_partial_as_func=True)


def has_exact_number_of_occurences(seq, *, val, count):
    """
    Check if seq has val appear exactly count times

    Parameters:
        seq(iterable): The sequence to be checked
        val : The hidden state to be counted
        count : The exact number

    Returns:
        bool: True iff satisfied
    """
    return seq.count(val) == count


def has_exact_number_of_occurences_constraint(*, val, count):
    return Constraint(
        func=lambda seq: has_exact_number_of_occurences(seq, val=val, count=count),
        partial_func=lambda T, seq: seq.count(val) <= count
        and seq.count(val) + T - len(seq) >= count,
    )


def appears_at_least_once(seq, val):
    """
    Checks if val appears at least once

    Parameters:
        seq(iterable): The sequence to be checked
        val : The hidden state to be counted

    Returns:
        bool: True iff satisfied
    """
    return has_minimum_number_of_occurences(seq, val=val, count=1)


def appears_at_least_once_constraint(val):
    # No partial because val could appear at the last time step
    return Constraint(func=lambda seq: appears_at_least_once(seq, val))


def does_not_occur(seq, val):
    """
    Checks if val does not occur

    Parameters:
        seq(iterable): The sequence to be checked
        val : The hidden state to check

    Returns:
        bool: True iff satisfied
    """
    return has_maximum_number_of_occurences(seq, val=val, count=0)


def does_not_occur_constraint(val):
    # No partial because val could appear at the last time step
    return Constraint(
        func=lambda seq: does_not_occur(seq, val), same_partial_as_func=True
    )


def fix_final_state(seq, val):
    """
    Requires that the final state of sequence is val

    Parameters:
        seq (iterable): The sequence to be checked
        val : The hidden state which must be the final state

    Returns:
        bool: True iff satisfied
    """
    # Return false if seq is empty
    if not seq:
        return False
    return seq[-1] == val


def fix_final_state_constraint(val):
    # No partial because it only involves the final state
    return Constraint(func=lambda seq: fix_final_state(seq, val))


def occurs_only_in_time_frame(seq, val, *, lower_t=None, upper_t=None):
    """
    Requires that val only occurs in seq[lower_t, upper_t]

    Parameters:
        seq (Iterable): The sequence to be checked
        val : The hidden state
        lower_t : The lower bound on time frame (inclusive)
                  If None we set this to 0
        upper_t : The upper bound on time frame (exclusive)
                  If None we set this to len(seq)

    Returns:
        bool: True iff satisfied
    """
    if lower_t is None:
        lower_t = 0
    if upper_t is None or upper_t > len(seq):
        upper_t = len(seq)

    return (seq[0:lower_t].count(val) == 0) and (
        seq[upper_t - 1 : len(seq)].count(val) == 0
    )


def occurs_only_in_time_frame_constraint(val, *, lower_t=None, upper_t=None):
    return Constraint(
        func=lambda seq: occurs_only_in_time_frame(
            seq, val, lower_t=lower_t, upper_t=upper_t
        ),
        same_partial_as_func=True,
    )


def occurs_at_least_once_in_time_frame(seq, val, *, lower_t=None, upper_t=None):
    """
    Requires that val only occurs at least once in seq[lower_t, upper_t]

    Parameters:
        seq (Iterable): The sequence to be checked
        val : The hidden state
        lower_t : The lower bound on time frame (inclusive)
                  If None we set this to 0
        upper_t : The upper bound on time frame (exclusive)
                  If None we set this to len(seq)

    Returns:
        bool: True iff satisfied
    """
    if lower_t is None:
        lower_t = 0
    if upper_t is None:
        upper_t = len(seq)

    return seq[lower_t:upper_t].count(val) >= 1


def occurs_at_least_once_in_time_frame_constraint(val, *, lower_t=None, upper_t=None):
    def partial_func(seq, val, lower_t, upper_t):
        if lower_t is None:
            lower_t = 0
        if upper_t is None:
            upper_t = len(seq)
        if len(seq) < upper_t or lower_t > len(seq):
            return True
        else:
            return occurs_at_least_once_in_time_frame(
                seq, val, lower_t=lower_t, upper_t=upper_t
            )

    return Constraint(
        func=lambda seq: occurs_at_least_once_in_time_frame(
            seq, val, lower_t=lower_t, upper_t=upper_t
        ),
        partial_func=lambda T, seq: partial_func(seq, val, lower_t, upper_t),
    )


# ------------------------------------------
# Modifications one can make to constraints
# ------------------------------------------


def or_constraints(constraints):
    """
    Takes in a list of constraints and returns a constraints that
    is true if at least on of the holds

    Parameters:
        constraints (iterable): List of Constraint objects

    Returns
        Constraint: Constraint satisfying desired properties
    """
    name = "or("
    for constraint in constraints:
        name = name + constraint.name + "_"
    name = name[:-1]  # Remove trailing underscore
    name += ")"

    def or_func(seq):
        for constraint in constraints:
            if constraint(seq):
                return True
        return False

    def or_partial_func(T, seq):
        for constraint in constraints:
            if constraint.partial_func(T, seq):
                return True
        return False

    return Constraint(func=or_func, partial_func=or_partial_func, name=name)


def xor_constraints(constraints):
    """
    Takes in a list of constraints and returns a constraints that
    is true if exactly one of them holds

    Parameters:
        constraints (iterable): List of Constraint objects

    Returns
        Constraint: Constraint satisfying desired properties
    """
    name = "xor("
    for constraint in constraints:
        name = name + constraint.name + "_"
    name = name[:-1]  # Remove trailing underscore
    name += ")"

    def xor_func(seq):
        at_least_one_true = False
        for constraint in constraints:
            if constraint(seq):
                if at_least_one_true:
                    return False
                at_least_one_true = True

        if at_least_one_true:
            return True
        return False

    # Not a strong partial_func, it's false only if all partial_funcs are false
    # Same as or_partial_func actually
    def xor_partial_func(T, seq):
        for constraint in constraints:
            if constraint.partial_func(T, seq):
                return True
            return False

    return Constraint(func=xor_func, partial_func=xor_partial_func, name=name)


def not_constraint(constraint):
    """
    Takes in a constraints and returns a constraints that
    is true iff the original constraint is false

    Note: If possible use built in functions, this makes it so that
    we can use the built in partial_func, whereas in the automated
    process we can't generate partial_func.

    Parameters:
        constraints (iterable): List of Constraint objects

    Returns
        Constraint: Constraint satisfying desired properties
    """
    name = "not(" + constraint.name + ")"

    def not_func(seq):
        return not constraint(seq)

    # No partial func for not_constraint

    return Constraint(func=not_func, name=name)


def and_constraints(constraints):
    """
    Takes in a list of constraints and returns a constraints that
    is true iff all constraints are true.
    Note: Probably don't use this other than for difficult modelling, you can just
    add multiple constraints to your statistical model

    Parameters:
        constraints (iterable): List of Constraint objects

    Returns
        Constraint: Constraint satisfying desired properties
    """
    name = "and("
    for constraint in constraints:
        name = name + constraint.name + "_"
    name = name[:-1]  # Remove trailing underscore
    name += ")"

    def and_func(seq):
        for constraint in constraints:
            if not constraint(seq):
                return False
        return True

    def and_partial_func(T, seq):
        for constraint in constraints:
            if not constraint.partial_func(T, seq):
                return False
        return True

    return Constraint(func=and_func, partial_func=and_partial_func, name=name)

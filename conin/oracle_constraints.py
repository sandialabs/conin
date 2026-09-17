import inspect
from conin.constraint import OracleConstraint, oracle_constraint_fn


def constraint_name():
    """Return the name of the calling function.

    This helper is used internally when naming generated constraints.
    """
    return inspect.stack()[1].function


@oracle_constraint_fn(same_partial_as_func=True)
def all_diff_constraint(seq):
    """Require every value in ``seq`` to be unique.

    Parameters
    ----------
    seq : iterable
        Sequence to check.

    Returns
    -------
    bool
        ``True`` if all elements are unique.
    """
    return len(seq) == len(set(seq))


def always_appears_before(seq, val1, val2):
    """Check whether all occurrences of ``val1`` precede ``val2``.

    Parameters
    ----------
    seq : iterable
        Sequence to check.
    val1
        Value that must appear before ``val2``.
    val2
        Value that must not appear before a later ``val1``.

    Returns
    -------
    bool
        ``True`` if no ``val1`` appears after the first ``val2``.
    """
    for index, x1 in enumerate(seq):
        if x1 == val2:
            for index2 in range(index + 1, len(seq)):
                if seq[index2] == val1:
                    return False
            return True
    return True


def always_appears_before_constraint(val1, val2):
    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        return always_appears_before(seq, val1, val2)

    return constraint


def appears_at_least_once_before_constraint(val1, val2):
    """Return a constraint requiring ``val1`` before the first ``val2``.

    Parameters
    ----------
    val1
        Value that must appear before ``val2``.
    val2
        Value whose first occurrence is checked.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        found_val1 = False

        for x in seq:
            if x == val1:
                found_val1 = True
            elif x == val2:
                return found_val1

        return True

    return constraint


def always_appears_after_constraint(val1, val2):
    """Return a constraint requiring ``val1`` to appear only after ``val2``.

    Parameters
    ----------
    val1
        Value that must appear after ``val2``.
    val2
        Value that must appear before ``val1``.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        return always_appears_before(seq, val2, val1)

    return constraint


def appears_at_least_once_after_constraint(val1, val2):
    """Return a constraint requiring ``val1`` after the first ``val2``.

    Parameters
    ----------
    val1
        Value that must appear after ``val2``.
    val2
        Value whose first occurrence is checked.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    # No partial here because val2 could appear at the very last time step
    @oracle_constraint_fn(name=constraint_name())
    def constraint(seq):
        for index1, x1 in enumerate(seq):
            if x1 == val2:
                for index2 in range(index1 + 1, len(seq)):
                    if seq[index2] == val1:
                        return True
                return False
        return True

    return constraint


@oracle_constraint_fn(same_partial_as_func=True)
def citation_constraint(seq):
    """Require each value to appear in a single contiguous block.

    For example, ``[1, 1, 2, 2, 7, 7]`` is feasible, but
    ``[1, 1, 2, 2, 7, 7, 2, 2]`` is not.

    Parameters
    ----------
    seq : iterable
        Sequence to check.

    Returns
    -------
    bool
        ``True`` if every value appears in one contiguous block.
    """
    for t1 in range(2, len(seq)):
        if seq[t1] != seq[t1 - 1]:
            for t2 in range(t1 - 2):
                if seq[t2] == seq[t1]:
                    return False
    return True


def has_minimum_number_of_occurences(seq, *, val, count):
    """Check whether ``val`` appears at least ``count`` times.

    The misspelling in this function name is preserved for API compatibility.

    Parameters
    ----------
    seq : iterable
        Sequence to check.
    val
        Value to count.
    count : int
        Minimum allowed number of occurrences.

    Returns
    -------
    bool
        ``True`` if ``val`` appears at least ``count`` times.
    """
    return seq.count(val) >= count


def has_minimum_number_of_occurences_constraint(*, val, count):
    """Return a constraint requiring at least ``count`` occurrences of ``val``.

    The misspelling in this function name is preserved for API compatibility.
    """
    return OracleConstraint(
        func=lambda seq: has_minimum_number_of_occurences(seq, val=val, count=count),
        partial_func=lambda T, seq: seq.count(val) + T - len(seq) >= count,
    )


def has_maximum_number_of_occurences(seq, *, val, count):
    """Check whether ``val`` appears at most ``count`` times.

    The misspelling in this function name is preserved for API compatibility.

    Parameters
    ----------
    seq : iterable
        Sequence to check.
    val
        Value to count.
    count : int
        Maximum allowed number of occurrences.

    Returns
    -------
    bool
        ``True`` if ``val`` appears at most ``count`` times.
    """
    return seq.count(val) <= count


def has_maximum_number_of_occurences_constraint(*, val, count):
    """Return a constraint requiring at most ``count`` occurrences of ``val``.

    The misspelling in this function name is preserved for API compatibility.
    """
    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        return has_maximum_number_of_occurences(seq, val=val, count=count)

    return constraint


def has_exact_number_of_occurences_constraint(*, val, count):
    """Return a constraint requiring exactly ``count`` occurrences of ``val``.

    The misspelling in this function name is preserved for API compatibility.
    """

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

    return OracleConstraint(
        func=lambda seq: has_exact_number_of_occurences(seq, val=val, count=count),
        partial_func=lambda T, seq: seq.count(val) <= count
        and seq.count(val) + T - len(seq) >= count,
    )


def appears_at_least_once_constraint(val):
    """Return a constraint requiring ``val`` to appear at least once.

    Parameters
    ----------
    val
        Value that must appear in the sequence.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    @oracle_constraint_fn(name=constraint_name())
    def constraint(seq):
        return has_minimum_number_of_occurences(seq, val=val, count=1)

    return constraint


def does_not_occur_constraint(val):
    """Return a constraint requiring ``val`` to be absent.

    Parameters
    ----------
    val
        Value that must not appear in the sequence.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        return has_maximum_number_of_occurences(seq, val=val, count=0)

    return constraint


def fix_final_state_constraint(val):
    """Return a constraint requiring the final sequence value to be ``val``.

    Parameters
    ----------
    val
        Required final value.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    # No partial because it only involves the final state
    @oracle_constraint_fn(name=constraint_name())
    def constraint(seq):
        if not seq:
            # Return false if seq is empty
            return False
        return seq[-1] == val

    return constraint


def occurs_only_in_time_frame_constraint(val, *, lower_t=None, upper_t=None):
    """Return a constraint limiting where ``val`` may occur.

    Parameters
    ----------
    val
        Value whose occurrence window is restricted.
    lower_t : int, optional
        Lower index before which ``val`` may not appear. Defaults to ``0``.
    upper_t : int, optional
        Upper window parameter. In the current implementation, ``val`` may not
        appear at indices ``upper_t - 1`` or later. Defaults to ``len(seq)``.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    @oracle_constraint_fn(name=constraint_name(), same_partial_as_func=True)
    def constraint(seq):
        lower = 0 if lower_t is None else lower_t
        upper = len(seq) if upper_t is None else upper_t

        return (seq[0:lower].count(val) == 0) and (
            seq[upper - 1 : len(seq)].count(val) == 0
        )

    return constraint


def occurs_at_least_once_in_time_frame_constraint(val, *, lower_t=None, upper_t=None):
    """Return a constraint requiring ``val`` inside a time window.

    Parameters
    ----------
    val
        Value that must appear in the time window.
    lower_t : int, optional
        Inclusive lower index. Defaults to ``0``.
    upper_t : int, optional
        Exclusive upper index. Defaults to ``len(seq)``.

    Returns
    -------
    OracleConstraint
        Constraint that evaluates sequences according to this rule.
    """

    def func(seq):
        lower = 0 if lower_t is None else lower_t
        upper = len(seq) if upper_t is None else upper_t

        return seq[lower:upper].count(val) >= 1

    def partial_func(T, seq):
        lower = 0 if lower_t is None else lower_t
        upper = len(seq) if upper_t is None else upper_t
        if len(seq) < upper or lower > len(seq):
            return True
        else:
            return func(seq)

    return OracleConstraint(func=func, partial_func=partial_func)


# ------------------------------------------
# Modifications one can make to constraints
# ------------------------------------------


def or_constraints(constraints):
    """Return a constraint satisfied when any input constraint is satisfied.

    Parameters
    ----------
    constraints : iterable of OracleConstraint
        Constraints to combine.

    Returns
    -------
    OracleConstraint
        Combined disjunctive constraint.
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

    return OracleConstraint(func=or_func, partial_func=or_partial_func, name=name)


def xor_constraints(constraints):
    """Return a constraint satisfied when exactly one input is satisfied.

    Parameters
    ----------
    constraints : iterable of OracleConstraint
        Constraints to combine.

    Returns
    -------
    OracleConstraint
        Combined exclusive-or constraint.
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

    return OracleConstraint(func=xor_func, partial_func=xor_partial_func, name=name)


def not_constraint(constraint):
    """Return a constraint that negates another constraint.

    The negated constraint cannot generally provide a strong ``partial_func``.

    Parameters
    ----------
    constraint : OracleConstraint
        Constraint to negate.

    Returns
    -------
    OracleConstraint
        Constraint whose value is ``not constraint(seq)``.
    """
    name = "not(" + constraint.name + ")"

    # No partial func for not_constraint
    @oracle_constraint_fn(name=name)
    def not_func(seq):
        return not constraint(seq)

    return not_func


def and_constraints(constraints):
    """Return a constraint satisfied when every input constraint is satisfied.

    For most models, adding each constraint separately is preferable. This helper
    is useful when constraints need to be composed before they are passed to a
    model.

    Parameters
    ----------
    constraints : iterable of OracleConstraint
        Constraints to combine.

    Returns
    -------
    OracleConstraint
        Combined conjunctive constraint.
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

    return OracleConstraint(func=and_func, partial_func=and_partial_func, name=name)

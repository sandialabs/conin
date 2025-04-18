import munch


def add_unknowns(hidden_list, *, num=1, token="__UNKNOWN__"):
    """
    Replace any value in hidden_vec which appear <= num times
    with token

    CLM -- As a hacky way to not lose information when doing this,
    you could run hidden_vec.extend(add_unknowns(hidden_vec)). This
    way you don't lose the fact that those values are actually known.

    Parameters:
        hidden_list: A list of lists of hidden states
        num: The cut of for when we replace a value with token
        token: What we replace hidden variables with

    Returns:
        list of lists: The modified version of hidden_vec
    """
    num_occurences = {}
    for hidden in hidden_list:
        for val in hidden:
            if val not in num_occurences:
                num_occurences[val] = 1
            else:
                num_occurences[val] += 1

    output = hidden_list
    for i, hidden in enumerate(output):
        for j, val in enumerate(hidden):
            if num_occurences[val] <= num:
                output[i][j] = token  # Update the value in the output list

    return output


def convert_to_simulations(*, hidden_list, observed_list):
    """
    Converts from hidden_list and observed_list to simulations format
    to be used in supervised learning for instance

    Parameters
        hidden_list: list of hidden
        observed_list: list of observations
    """
    simulations = []
    for i, hidden in enumerate(hidden_list):
        res = munch.Munch(hidden=hidden, index=i, observed=observed_list[i])
        simulations.append(res)
    return simulations

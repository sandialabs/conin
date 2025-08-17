import copy
import numpy as np
from conin.exceptions import InvalidInputError


class Util:
    @staticmethod
    def sample_from_vec(probabilities):
        val = np.random.uniform()
        temp = 0
        for i in range(len(probabilities)):
            temp += probabilities[i]
            if temp >= val:
                return i

    @staticmethod
    def normalize_vector(vec):
        total = sum(vec)
        if total == 0:
            return [1 / len(vec) for _ in vec]
        return [float(x) / total for x in vec]  # Convert to float explicitly

    @staticmethod
    def normalize_matrix(mat):
        return [
            list(map(float, Util.normalize_vector(vec))) for vec in mat
        ]  # Convert each normalized vector to a list of floats

    @staticmethod
    def normalize_dictionary(dict):
        sum = 0
        for value in dict.values():
            sum += value

        normalized_dict = copy.copy(dict)
        if sum != 0:
            for key in dict:
                normalized_dict[key] /= sum
        else:
            for key in dict:
                normalized_dict[key] = 1 / len(dict)
        return normalized_dict

    @staticmethod
    def normalize_2d_dictionary(dict):
        sums = {}
        len = {}
        for (key1, key2), val in dict.items():
            if key1 not in sums:
                sums[key1] = 0
                len[key1] = 0
            sums[key1] += val
            len[key1] += 1

        normalized_dict = {}
        for (key1, key2), val in dict.items():
            if sums[key1] != 0:
                normalized_val = val / sums[key1]
            else:
                normalized_val = 1 / len[key1]
            normalized_dict[(key1, key2)] = normalized_val
        return normalized_dict


class try_import:

    def __init__(self):
        self.success = False

    def __bool__(self):
        return self.success

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("HERE", exc_type, print(type(exc_type)), exc_value)
        if exc_type is ImportError or exc_type is ModuleNotFoundError:
            # Catch import errors
            return True
        elif exc_type is not None:
            # Propogate other exceptions
            return False

        # No import errors, so return 
        self.success = True
        return True

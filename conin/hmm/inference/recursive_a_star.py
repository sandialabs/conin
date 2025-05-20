class Recursive_Heap_Item:
    __slots__ = ["_priority", "_last_element", "_length", "_constraint_data"]

    def __init__(self, *, priority, last_element, length, constraint_data):
        """
        Initializes the object

        Parameters:
        priority (float): Priority in queue
        last_element (in H): Last element in sequence (must be hashable)
        length (int): Length of sequence (>= 0)
        constraint_data (any): Data needed to check constraint feasibility (must be hashable)
        """
        # Priority is a number
        if not isinstance(priority, (int, float)):
            raise TypeError(f"priority in Recursive_Heap_item must be a float.")

        # last_element is hashable
        try:
            hash(last_element)
        except:
            raise TypeError(f"last_element in Recursive_Heap_Item must be hashable.")

        # length is a positive integer
        if not isinstance(length, int):
            raise TypeError(f"length in Recursive_Heap_Item must be a int.")
        else:
            if length <= 0:
                raise TypeError(f"length in Recursive_Heap_Item must be >= 0.")

        # constraint_data is hashable
        try:
            hash(constraint_data)
        except:
            raise TypeError(f"constraint_data must be hashable.")

        self._priority = priority
        self._last_element = last_element
        self._length = length
        self._constraint_data = constraint_data

    # Getters
    @property
    def priority(self):
        return self._priority

    @property
    def last_element(self):
        return self._last_element

    @property
    def length(self):
        return self._length

    @property
    def constraint_data(self):
        return self._constraint_data

    def __lt__(self, y):
        """
        Only depends on priority
        """
        return self.priority < y.priority

    def __eq__(self, y):
        """
        Depends on all elements (not just priority)
        """
        return (
            (self.priority == y.priority)
            and (self.last_element == y.last_element)
            and (self.length == y.length)
            and (self.constraint_data == y.constraint_data)
        )

    def __hash__(self):
        """
        Just put everything in a tuple and take the hash of that
        """
        return hash(
            (self.priority, self.last_element, self.length, self.constraint_data)
        )

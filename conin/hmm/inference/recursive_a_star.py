import heapq


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

    def get_identifier(self):
        """
        Returns everything but the priority in a tuple.
        Used in Unique_Heap
        """
        return (self.last_element, self.length, self.constraint_data)


class Unique_Heap:
    """
    This is used in our tailored A* algorithm
    We consider to elements "equal" if their sequences end with the same
    value and the have the same data associated with them.
    """

    def __init__(self):
        self.__heap = []
        self.__unique_map = {}  # Maps identifier to Recursive_Heap_Item

    def add(self, item):
        """
        Adds an item only if it doesn't already exist
        """
        identifier = item.get_identifier()

        # Check if the element is already in the heap
        if identifier in self.__unique_map:
            existing_item = self.__unique_map[identifier]

            if item.priority < existing_item.priority:
                self.__heap.remove(existing_item)
                heapq.heapify(self.__heap)  # Re-heapify after removal
                heapq.heappush(self.__heap, item)
                self.__unique_map[identifier] = item
        else:
            # Add the identifier to the map and the item to the heap
            self.__unique_map[identifier] = item
            heapq.heappush(self.__heap, item)

    def pop(self):
        """
        Pops the item with the smallest priority
        """
        while self.__heap:
            item = heapq.heappop(self.__heap)

            # Update our unique_map
            identifier = item.get_identifier()
            del self.__unique_map[identifier]

            return item
        raise IndexError("pop from an empty heap")

    def __len__(self):
        """
        Length of the heapq
        """
        return len(self.__heap)

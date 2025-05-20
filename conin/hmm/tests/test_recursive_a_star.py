import pytest

from conin.hmm.inference.recursive_a_star import *


@pytest.fixture
def hmm():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    hmm = HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


@pytest.fixture
def heap_item():
    return Recursive_Heap_Item(
        priority=1, last_element="h0", length=10, constraint_data=(1, 2)
    )


class Test_Heap_Item:
    def test_init(self, heap_item):

        # Priority is a float
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority="a", last_element="h0", length=10, constraint_data=(1, 2)
            )

        # Last element is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element=["h0"], length=10, constraint_data=(1, 2)
            )

        # Length > 0
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=-10, constraint_data=(1, 2)
            )
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=0, constraint_data=(1, 2)
            )

        # constraint_data is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=10, constraint_data=[1, 2]
            )

    def test_getters(self, heap_item):
        assert heap_item.priority == 1
        assert heap_item.last_element == "h0"
        assert heap_item.length == 10
        assert heap_item.constraint_data == (1, 2)

    def test_setters(self, heap_item):
        # heap_item should be immutable
        with pytest.raises(AttributeError):
            heap_item.priority = 2
        with pytest.raises(AttributeError):
            heap_item.last_element = "h1"
        with pytest.raises(AttributeError):
            heap_item.length = 9
        with pytest.raises(AttributeError):
            heap_item.constraint_data = (2, 3)

    def test_lt(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h1", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=10, constraint_data=(1, 2)
        )

        assert not heap_item < x2
        assert heap_item < x3
        assert x2 < x3

    def test_eq(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=2, last_element="h1", length=10, constraint_data=(1, 2)
        )

        assert heap_item == x2
        assert not heap_item == x3
        assert not x2 == x3
        assert not heap_item == x4
        assert not x2 == x4
        assert not x3 == x4

    def test_hash(self, heap_item):
        hash(heap_item)

    def test_get_identifier(self, heap_item):
        assert heap_item.get_identifier() == ("h0", 10, (1, 2))


class Test_Unique_Heap:
    def test_init(self):
        A = Unique_Heap()

    def test_add(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=4, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=2, last_element="h1", length=10, constraint_data=(1, 2)
        )

        A = Unique_Heap()
        A.add(heap_item)
        assert len(A) == 1
        A.add(x2)
        assert len(A) == 1
        A.add(x3)
        assert len(A) == 2
        A.add(heap_item)
        assert len(A) == 2
        A.add(x4)
        assert len(A) == 3
        A.add(x2)
        assert len(A) == 3
        A.add(x3)
        assert len(A) == 3
        A.add(x4)
        assert len(A) == 3
        assert set(A._Unique_Heap__heap) == {x2, x3, x4}

    def test_pop(self, heap_item):
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=4, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=3, last_element="h1", length=10, constraint_data=(1, 2)
        )

        A = Unique_Heap()
        A.add(heap_item)
        A.add(x4)
        A.add(x3)
        while A:
            A.pop()
        assert len(A) == 0

        A.add(heap_item)
        A.add(x4)
        A.add(x3)

        x = A.pop()
        assert x == heap_item
        x = A.pop()
        assert x == x3
        x = A.pop()
        assert x == x4
        assert len(A) == 0

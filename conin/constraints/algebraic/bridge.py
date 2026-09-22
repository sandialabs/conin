"""
Bridge layer connecting conin variables to smoek expression system.

This module provides the ConinVarNode class that extends smoek's ExprLeaf,
allowing conin model.V() variable references to participate in smoek's
algebraic expression building.
"""

from conin.util import try_import

with try_import() as smoek_available:
    from smoek.core.expr.nodes import ExprLeaf
if not smoek_available:
    class ExprLeaf(objecti): pass

class ConinVarNode(ExprLeaf):
    """
    Bridge between conin model.V() and smoek expression system.

    This class represents a reference to a conin variable in a smoek expression tree.
    It inherits all operator overloading from smoek.ExprLeaf, enabling natural
    algebraic syntax like: model.V("A", 0) + model.V("B", 1) <= 10

    Attributes:
        node: Variable node name (e.g., "A", "B")
        state: State value for the variable
        time: Optional time index for temporal models (HMM/DBN)

    Examples:
        >>> # Binary Network/Markov Network (2 args)
        >>> var = ConinVarNode("A", 0)
        >>> var.node, var.state
        ('A', 0)

        >>> # Hidden Markov Model (3 args with time)
        >>> var = ConinVarNode("X", 1, time=5)
        >>> var.node, var.state, var.time
        ('X', 1, 5)
    """

    def __init__(self, node, state, time=None):
        """
        Initialize a conin variable node.

        Args:
            node: Variable node name
            state: State value
            time: Optional time index (for HMM/DBN models)
        """
        super().__init__()
        self.node = node
        self.state = state
        self.time = time

    def __repr__(self):
        if self.time is None:
            return f"ConinVarNode({self.node!r}, {self.state!r})"
        else:
            return f"ConinVarNode({self.node!r}, {self.state!r}, time={self.time!r})"

    def __str__(self):
        if type(self.node) is str:
            index = f'"{self.node}"'
        else:
            index = self.node
        if type(self.state) is str:
            state = f'"{self.state}"'
        else:
            state = self.state
        if self.time is None:
            return f"m_.V({index}, {state})"
        else:
            return f"m_.V({index}, {self.time}, {state})"

    def to_string(self):
        """Return string representation for smoek walkers."""
        return str(self)

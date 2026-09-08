"""Parse MVR formulas and lower them through existing primitives and operators."""

from __future__ import annotations

import dataclasses
import re

from typing import Any

from conin.constraint import mvr_constraint_fn
from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr_constraints import (
    mvr_constant,
    mvr_current_sequencelist,
    mvr_current_state,
    mvr_current_transition,
    mvr_holdingtime,
    mvr_jump,
    mvr_jumpcounts,
    mvr_regex,
    mvr_withinbounds,
)
from conin.hidden_markov_model.mvr_operators import (
    mvr_already_satisfied,
    mvr_and,
    mvr_concatenate,
    mvr_concatenate_prefix,
    mvr_count,
    mvr_kfold_product,
    mvr_kfold_product_prefix,
    mvr_kleene_closure,
    mvr_kleene_closure_prefix,
    mvr_not,
    mvr_not_yet,
    mvr_or,
    mvr_precedence,
    mvr_reverse,
    mvr_sattime,
    mvr_setdiff,
    mvr_timerange,
)

# Temporal words name mvr_precedence; "cat" names the regular-language operator.
_PRECEDENCE_RELATION = {
    "then": "<",
    "before": "<",
    "after": ">",
    "at or before": "<=",
    "at or after": ">=",
}

_UNARY_OPERATORS = {
    "not": mvr_not,
    "never": mvr_not_yet,
    "reach": mvr_already_satisfied,
    "first": mvr_sattime,
    "reverse": mvr_reverse,
}

_BINARY_OPERATORS = {"and": mvr_and, "or": mvr_or, "but not": mvr_setdiff}

_KEYWORDS = frozenset(
    {
        "after",
        "and",
        "at",
        "before",
        "but",
        "cat",
        "count",
        "false",
        "first",
        "hold",
        "in",
        "is",
        "jump",
        "jumps",
        "match",
        "never",
        "not",
        "or",
        "over",
        "reach",
        "reverse",
        "seq",
        "then",
        "true",
        "within",
        "between",
    }
)

_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<qlabel><[^<>\s]*>)
  | (?P<string>"[^"]*")
  | (?P<number>-?\d+(?:\.\d+)?)
  | (?P<arrow>->)
  | (?P<relop>>=|<=|==|>|<)
  | (?P<word>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<punct>[(),\[\]{}:+])
    """,
    re.VERBOSE,
)


@dataclasses.dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    column: int


def _tokenize(formula: str) -> list[_Token]:
    """Tokenize a formula, reporting the column of the first bad character."""
    tokens = []
    position = 0

    while position < len(formula):
        match = _TOKEN_RE.match(formula, position)

        if match is None:
            raise _formula_error(formula, position, "unexpected character")

        kind = match.lastgroup
        text = match.group()
        position = match.end()

        if kind == "space":
            continue

        if kind == "word" and text in _KEYWORDS:
            kind = "keyword"

        tokens.append(_Token(kind=kind, text=text, column=match.start()))

    tokens.append(_Token(kind="end", text="", column=len(formula)))

    return tokens


def _formula_error(formula: str, column: int, message: str) -> InvalidInputError:
    """Point a caret at the offending column."""
    return InvalidInputError(
        f"{message} at column {column} of formula\n  {formula}\n  {' ' * column}^"
    )


# ------------------------------------------------------------------
# Syntax tree
# ------------------------------------------------------------------
# One node per production; "column" lets a lowering error point at the source.


@dataclasses.dataclass(frozen=True, slots=True)
class _States:
    labels: tuple[str, ...]
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Constant:
    value: bool
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Jump:
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Transition:
    source: str
    target: str
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Sequence:
    labels: tuple[str, ...]
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Holding:
    k: int
    labels: tuple[str, ...] | None
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _JumpCounts:
    condition: str
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _WithinBounds:
    bounds: Any
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Regex:
    pattern: str
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Unary:
    op: str
    operand: Any
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Binary:
    op: str
    left: Any
    right: Any
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Count:
    operand: Any
    condition: str
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Repeat:
    op: str
    operand: Any
    k: int | None
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class _Window:
    operand: Any
    time_range: tuple[int, int]
    column: int


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


class _Parser:
    """Recursive-descent parser with one method per precedence level."""

    def __init__(self, formula: str):
        self.formula = formula
        self.tokens = _tokenize(formula)
        self.position = 0

    # -- token helpers ---------------------------------------------

    @property
    def token(self) -> _Token:
        return self.tokens[self.position]

    def advance(self) -> _Token:
        token = self.token
        self.position += 1
        return token

    def at_label(self) -> bool:
        return self.token.kind in ("word", "qlabel", "number")

    def at(self, *texts: str) -> bool:
        return self.token.text in texts and self.token.kind in (
            "keyword",
            "punct",
            "arrow",
            "relop",
        )

    def at_words(self, *words: str) -> bool:
        """Check for a multi-word keyword such as 'at or before'."""
        for offset, word in enumerate(words):
            index = self.position + offset

            if index >= len(self.tokens) or self.tokens[index].text != word:
                return False

        return True

    def expect(self, text: str) -> _Token:
        if self.token.text != text:
            raise self.error(f"expected {text!r}, found {self.token.text or 'end'!r}")

        return self.advance()

    def error(self, message: str) -> InvalidInputError:
        return _formula_error(self.formula, self.token.column, message)

    # -- levels ----------------------------------------------------

    def parse(self) -> Any:
        node = self.parse_or()

        if self.token.kind != "end":
            raise self.error(f"unexpected {self.token.text!r}")

        return node

    def parse_or(self) -> Any:
        node = self.parse_and()

        while self.at("or"):
            column = self.advance().column
            node = _Binary("or", node, self.parse_and(), column)

        return node

    def parse_and(self) -> Any:
        node = self.parse_setdiff()

        while self.at("and"):
            column = self.advance().column
            node = _Binary("and", node, self.parse_setdiff(), column)

        return node

    def parse_setdiff(self) -> Any:
        # Below concatenation, with the other set operators. See CLAUDE.md.
        node = self.parse_window()

        while self.at_words("but", "not"):
            column = self.token.column
            self.position += 2
            node = _Binary("but not", node, self.parse_window(), column)

        return node

    def parse_window(self) -> Any:
        node = self.parse_temporal()

        # Integer operands are what let this consume the trailing "and".
        if self.at("between"):
            column = self.advance().column
            start = self.parse_integer()
            self.expect("and")
            end = self.parse_integer()

            return _Window(node, (start, end), column)

        if self.at("over"):
            column = self.advance().column
            self.expect("[")
            start = self.parse_integer()
            self.expect(",")
            end = self.parse_integer()
            self.expect("]")

            return _Window(node, (start, end), column)

        return node

    def parse_temporal(self) -> Any:
        node = self.parse_cat()
        operands = [node]
        relations = []
        column = self.token.column

        while True:
            relation = self.match_temporal()

            if relation is None:
                break

            relations.append(relation)
            operands.append(self.parse_cat())

        if not relations:
            return node

        if len(set(relations)) > 1:
            raise _formula_error(
                self.formula,
                column,
                "a chain of temporal operators must use one relation throughout; "
                "parenthesize to mix them",
            )

        # Chained like Python comparisons: "A then B then C" is pairwise.
        chain = [
            _Binary(relations[i], operands[i], operands[i + 1], column)
            for i in range(len(relations))
        ]
        node = chain[0]

        for step in chain[1:]:
            node = _Binary("and", node, step, column)

        return node

    def match_temporal(self) -> str | None:
        """Consume a temporal operator and return its relation, if present."""
        for text, relation in _PRECEDENCE_RELATION.items():
            words = text.split()
            if self.at_words(*words):
                self.position += len(words)

                return relation

        return None

    def parse_cat(self) -> Any:
        node = self.parse_unary()

        while self.at("cat"):
            column = self.advance().column
            node = _Binary("cat", node, self.parse_unary(), column)

        return node

    def parse_unary(self) -> Any:
        if self.at(*_UNARY_OPERATORS):
            token = self.advance()

            return _Unary(token.text, self.parse_unary(), token.column)

        return self.parse_postfix()

    def parse_postfix(self) -> Any:
        node = self.parse_atom()

        while True:
            if self.at("+"):
                column = self.advance().column
                node = _Repeat("kleene", node, None, column)
            elif self.at("{"):
                column = self.advance().column
                k = self.parse_integer()
                self.expect("}")
                node = _Repeat("kfold", node, k, column)
            elif self.at("["):
                # "A[3]" is sugar for "count(A) 3".
                column = self.advance().column
                k = self.parse_integer()
                self.expect("]")
                node = _Count(node, str(k), column)
            else:
                return node

    # -- atoms -----------------------------------------------------

    def parse_atom(self) -> Any:
        token = self.token

        if token.kind == "end":
            raise self.error("expected an expression, found end of formula")

        if self.at("true", "false"):
            self.advance()

            return _Constant(token.text == "true", token.column)

        if self.at("jump"):
            self.advance()

            return _Jump(token.column)

        if self.at("seq"):
            self.advance()
            self.expect("(")
            labels = self.parse_label_list()
            self.expect(")")

            return _Sequence(labels, token.column)

        if self.at("hold"):
            self.advance()
            k = self.parse_integer()
            labels = None

            if self.at("in"):
                self.advance()
                if self.at("("):
                    self.advance()
                    labels = self.parse_label_list()
                    self.expect(")")
                else:
                    labels = (self.parse_label(),)

            return _Holding(k, labels, token.column)

        if self.at("jumps"):
            self.advance()

            return _JumpCounts(self.parse_condition(), token.column)

        if self.at("within"):
            self.advance()

            return _WithinBounds(self.parse_bounds(), token.column)

        if self.at("match"):
            self.advance()

            if self.token.kind != "string":
                raise self.error('match expects a quoted pattern, e.g. match "<a><b>*"')

            return _Regex(self.advance().text[1:-1], token.column)

        if self.at("count"):
            self.advance()
            self.expect("(")
            operand = self.parse_or()
            self.expect(")")

            return _Count(operand, self.parse_condition(), token.column)

        if self.at("("):
            return self.parse_parenthesized()

        if self.at_label():
            label = self.parse_label()

            if self.at("->"):
                self.advance()

                return _Transition(label, self.parse_label(), token.column)

            return _States((label,), token.column)

        raise self.error(f"unexpected {token.text!r}")

    def parse_parenthesized(self) -> Any:
        """Parse a state group or a parenthesized expression."""
        open_token = self.expect("(")
        start = self.position
        labels = self.try_parse_label_group_body()

        if labels is not None:
            return _States(labels, open_token.column)

        self.position = start
        node = self.parse_or()
        self.expect(")")

        return node

    def try_parse_label_group_body(self) -> tuple[str, ...] | None:
        """Read 'A, B)' as a group; leave '(A)' to expression parsing."""
        labels = []

        while True:
            if not self.at_label():
                return None

            labels.append(self.parse_label())

            if self.at(","):
                self.advance()
                continue

            if self.at(")") and len(labels) > 1:
                self.advance()

                return tuple(labels)

            return None

    def parse_label(self) -> str:
        token = self.token

        if self.at_label():
            self.advance()

            return token.text[1:-1] if token.kind == "qlabel" else token.text

        raise self.error(
            f"expected a hidden state label, found {token.text or 'end'!r}; write a "
            "label that is also a keyword as <label>"
        )

    def parse_label_list(self) -> tuple[str, ...]:
        labels = [self.parse_label()]

        while self.at(","):
            self.advance()
            labels.append(self.parse_label())

        return tuple(labels)

    def parse_integer(self) -> int:
        token = self.token

        if token.kind != "number" or "." in token.text:
            raise self.error(f"expected an integer, found {token.text or 'end'!r}")

        self.advance()

        return int(token.text)

    def parse_number(self) -> float:
        token = self.token

        if token.kind != "number":
            raise self.error(f"expected a number, found {token.text or 'end'!r}")

        self.advance()

        return float(token.text) if "." in token.text else int(token.text)

    def parse_condition(self) -> str:
        """Parse a condition in mvr_count's mini-language."""
        if self.at("in"):
            self.advance()

            if not self.at("[", "("):
                raise self.error("expected a range like [1,3] or (1,3]")

            left = self.advance().text
            lower = self.parse_integer()
            self.expect(",")
            upper = self.parse_integer()
            self.expect("]")

            return f"{left}{lower},{upper}]"

        if self.token.kind == "relop":
            operator = self.advance().text

            if operator == "==":
                return str(self.parse_integer())

            return f"{operator}{self.parse_integer()}"

        if self.at("is"):
            self.advance()

            return str(self.parse_integer())

        if self.token.kind == "number":
            return str(self.parse_integer())

        raise self.error(
            "expected a count condition: an integer, 'in [1,3]', or a comparison "
            "like '>= 2'"
        )

    def parse_bounds(self) -> Any:
        """Parse a bounds pair or a map of time to bounds pairs."""
        if self.at("{"):
            self.advance()
            bounds = {}

            while not self.at("}"):
                time = self.parse_integer()
                self.expect(":")
                bounds[time] = self.parse_bounds_pair()

                if self.at(","):
                    self.advance()

            self.expect("}")

            return bounds

        return self.parse_bounds_pair()

    def parse_bounds_pair(self) -> list:
        self.expect("[")
        lower = self.parse_number()
        self.expect(",")
        upper = self.parse_number()
        self.expect("]")

        return [lower, upper]


# ------------------------------------------------------------------
# Lowering
# ------------------------------------------------------------------


def _resolve_labels(hidden_markov_model, labels, formula, column):
    """Resolve label text against the model's hidden states by str()."""
    hidden_states = list(getattr(hidden_markov_model, "hidden_states", None) or [])
    by_text = {str(h): h for h in hidden_states}

    if len(by_text) != len(hidden_states):
        raise InvalidInputError(
            "hidden states must have distinct string forms, since a formula names "
            f"them as text: {sorted(str(h) for h in hidden_states)}"
        )

    unknown = [label for label in labels if label not in by_text]

    if unknown:
        raise _formula_error(
            formula,
            column,
            f"{unknown[0]!r} is not a hidden state of hidden_markov_model "
            f"(states are {sorted(by_text)})",
        )

    return [by_text[label] for label in labels]


def _lower(node, hidden_markov_model, formula):
    """Lower one syntax node through the MVR algebra."""
    labels = lambda values: _resolve_labels(
        hidden_markov_model, values, formula, node.column
    )

    if isinstance(node, _States):
        return mvr_current_state(hidden_markov_model, set(labels(node.labels)))

    if isinstance(node, _Constant):
        return mvr_constant(hidden_markov_model, node.value)

    if isinstance(node, _Jump):
        return mvr_jump(hidden_markov_model)

    if isinstance(node, _Transition):
        source, target = labels((node.source, node.target))

        return mvr_current_transition(hidden_markov_model, [(source, target)])

    if isinstance(node, _Sequence):
        return mvr_current_sequencelist(
            hidden_markov_model, [list(labels(node.labels))]
        )

    if isinstance(node, _Holding):
        states = None if node.labels is None else set(labels(node.labels))

        return mvr_holdingtime(hidden_markov_model, node.k, states)

    if isinstance(node, _JumpCounts):
        return mvr_jumpcounts(hidden_markov_model, node.condition)

    if isinstance(node, _WithinBounds):
        return mvr_withinbounds(hidden_markov_model, node.bounds)

    if isinstance(node, _Regex):
        return mvr_regex(hidden_markov_model, node.pattern)

    if isinstance(node, (_Unary, _Count, _Repeat, _Window)):
        operand = _lower(node.operand, hidden_markov_model, formula)

    if isinstance(node, _Unary):
        return _UNARY_OPERATORS[node.op](operand)

    if isinstance(node, _Count):
        return mvr_count(operand, node.condition)

    if isinstance(node, _Repeat):
        if node.op == "kleene":
            builder = (
                mvr_kleene_closure_prefix if operand.prefix else mvr_kleene_closure
            )

            return builder(operand)

        builder = mvr_kfold_product_prefix if operand.prefix else mvr_kfold_product

        return builder(operand, node.k)

    if isinstance(node, _Window):
        # Applied last and never in place. See CLAUDE.md.
        return mvr_timerange(operand, list(node.time_range), inplace=False)

    if isinstance(node, _Binary):
        left = _lower(node.left, hidden_markov_model, formula)
        right = _lower(node.right, hidden_markov_model, formula)

        if node.op in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[node.op]([left, right])

        if node.op == "cat":
            builder = (
                mvr_concatenate_prefix
                if left.prefix and right.prefix
                else mvr_concatenate
            )

            return builder([left, right])

        return mvr_precedence([left, right], node.op)

    raise InvalidInputError(f"unsupported formula node: {type(node).__name__}")


def _conjuncts(node) -> list:
    """Split a top-level conjunction, distributing its window to each part."""
    if isinstance(node, _Binary) and node.op == "and":
        return _conjuncts(node.left) + _conjuncts(node.right)

    if isinstance(node, _Window):
        return [
            _Window(part, node.time_range, node.column)
            for part in _conjuncts(node.operand)
        ]

    return [node]


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------


def _named_parts(formula: str, name: str = None) -> list:
    """Parse a formula into (conjunct, name) pairs."""
    if not isinstance(formula, str):
        raise InvalidInputError("formula must be a string")

    parts = _conjuncts(_Parser(formula).parse())
    base = formula if name is None else name

    return [
        (part, base if len(parts) == 1 else f"{base}[{index}]")
        for index, part in enumerate(parts)
    ]


def _build_named(part, hidden_markov_model, formula, name):
    """Name the lowered conjunct after operators have dropped _name."""
    mvr = _lower(part, hidden_markov_model, formula)
    mvr.name = name

    return mvr


def _builder(part, formula, name):
    """Close over one conjunct without changing the factory's arity."""

    def build(hidden_markov_model):
        return _build_named(part, hidden_markov_model, formula, name)

    return build


def build_mvr(hidden_markov_model, formula: str, *, name: str = None) -> list:
    """Build a list of MVRs, one per top-level conjunct, for MVR_CHMM.

    Names use the formula (or ``name``), indexed when there are multiple results.
    Invalid syntax, unknown states, and invalid arguments raise InvalidInputError.
    """
    return [
        _build_named(part, hidden_markov_model, formula, part_name)
        for part, part_name in _named_parts(formula, name)
    ]


def build_mvr_functor(formula: str, *, name: str = None) -> list:
    """Return MVRConstraint factories for ConstrainedHiddenMarkovModel.

    Parse immediately; lower when an HMM is supplied. Naming follows build_mvr
    and applies to both factories and their results.
    """
    return [
        mvr_constraint_fn(name=part_name)(_builder(part, formula, part_name))
        for part, part_name in _named_parts(formula, name)
    ]

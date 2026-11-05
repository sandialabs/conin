"""Read and write Bayesian networks in the BIF (Bayesian Interchange Format).

The reader accepts the text BIF dialect used by the bnlearn repository and
JavaBayes::

    network asia {
    }
    variable smoke {
      type discrete [ 2 ] { yes, no };
    }
    probability ( smoke ) {
      table 0.5, 0.5;
    }
    probability ( lung | smoke ) {
      (yes) 0.1, 0.9;
      (no) 0.01, 0.99;
    }

``//`` and ``/* */`` comments and ``property ...;`` statements are ignored.
Names and states may be quoted (``"a b"``); unquoted names may contain
spaces (consecutive words are joined with one space). A state list without
commas is split at whitespace. In a ``probability`` header without ``|``, the
first name is the node and the others are its parents. A ``table`` entry
lists every probability, with the node's state varying slowest and the last
parent's state fastest (the JavaBayes order); a ``default`` entry
gives the distribution for every parent configuration without its own row.
A name in a ``probability`` header that matches no declared variable is
matched case-insensitively.
"""

import re

from conin.exceptions import InvalidInputError, ModelFileError
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD

_PUNCTUATION = "{}()[]|,;"

# Comments, quoted strings, punctuation, and words (anything else).
_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<comment>//[^\n]*|/\*.*?\*/)
    | "(?P<quoted>[^"]*)"
    | (?P<punct>[{}()\[\]|,;])
    | (?P<word>(?:[^\s{}()\[\]|,;"/]|/(?![/*]))+)
    """,
    re.VERBOSE | re.DOTALL,
)

# Names and states that the writer leaves unquoted.
_PLAIN_RE = re.compile(r"[A-Za-z0-9_.+\-]+")

_KEYWORDS = {"network", "variable", "probability", "property", "table", "default"}


class _Token:
    __slots__ = ("kind", "text", "line")

    def __init__(self, kind, text, line):
        self.kind = kind  # "word", "quoted" or "punct"
        self.text = text
        self.line = line

    def is_punct(self, text):
        return self.kind == "punct" and self.text == text

    def is_word(self, text):
        return self.kind == "word" and self.text == text


def _tokenize(text):
    tokens = []
    line = 1
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            # Only an unterminated quote or comment gets here.
            what = "comment" if text.startswith("/*", pos) else "quoted string"
            raise ModelFileError(f"Unterminated {what} on line {line} of the BIF data")
        kind = m.lastgroup
        if kind in ("word", "quoted", "punct"):
            tokens.append(_Token(kind, m.group(kind), line))
        line += m.group(0).count("\n")
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def error(self, message, token=None):
        if token is None:
            token = self.peek()
        where = f"line {token.line}" if token is not None else "the end of the BIF data"
        return ModelFileError(f"{message} at {where}")

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self, what):
        token = self.peek()
        if token is None:
            raise self.error(f"Expected {what}")
        self.pos += 1
        return token

    def expect(self, punct):
        token = self.next(f"'{punct}'")
        if not token.is_punct(punct):
            raise self.error(f"Expected '{punct}', got {token.text!r}", token)

    def accept(self, punct):
        token = self.peek()
        if token is not None and token.is_punct(punct):
            self.pos += 1
            return True
        return False

    def words(self, stop):
        """The non-punctuation tokens before the next token in ``stop``."""
        result = []
        while True:
            token = self.peek()
            if token is None:
                raise self.error(f"Expected one of {' '.join(stop)}")
            if token.kind == "punct":
                if token.text in stop:
                    return result
                raise self.error(f"Unexpected {token.text!r}", token)
            result.append(token)
            self.pos += 1

    def name(self, stop, what):
        """One name: a quoted string or words joined with spaces."""
        token = self.peek()
        words = self.words(stop)
        if not words:
            raise self.error(f"Expected {what}", token)
        if len(words) > 1 and any(w.kind == "quoted" for w in words):
            raise self.error(f"Unexpected text after a quoted {what}", words[1])
        return " ".join(w.text for w in words)

    def names(self, stop, what, split_words=True):
        """Names separated by commas, or by whitespace if there is no comma
        and ``split_words`` is true."""
        result = [self.words(stop + ",")]
        while self.accept(","):
            result.append(self.words(stop + ","))
        if len(result) == 1 and (split_words or not result[0]):
            return [w.text for w in result[0]]
        for words in result:
            if not words:
                raise self.error(f"Expected {what}")
        return [" ".join(w.text for w in words) for words in result]

    def skip_statement(self):
        """Skip to the end of a ``property ...;`` statement."""
        while not self.next("';'").is_punct(";"):
            pass

    def numbers(self):
        """Numbers separated by commas or whitespace, up to ``;``."""
        values = []
        while not self.accept(";"):
            if self.accept(","):
                continue
            token = self.next("a probability or ';'")
            try:
                values.append(float(token.text))
            except ValueError:
                raise self.error(
                    f"Expected a probability, got {token.text!r}", token
                ) from None
        return values


def load_bif(filename=None, string=None):
    """Load a Bayesian network from a BIF file or string.

    Parameters
    ----------
    filename : str, optional
        Path of a ``.bif`` file.
    string : str, optional
        The BIF text (used when ``filename`` is not given).

    Returns
    -------
    DiscreteBayesianNetwork
        The nodes and states are strings, in the order of the file.

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    ModelFileError
        If the data is malformed or does not describe a valid Bayesian
        network.
    NotImplementedError
        For a variable whose type is not ``discrete``.
    """
    if filename is not None:
        with open(filename, encoding="utf-8") as INPUT:
            string = INPUT.read()
    if string is None:
        raise InvalidInputError("load_bif needs a filename or a string")
    try:
        return _load_bif(string)
    except ModelFileError:
        raise
    except InvalidInputError as e:
        raise ModelFileError(f"Invalid BIF data: {e}") from e


def _load_bif(text):
    parser = _Parser(_tokenize(text))
    states = {}
    lowercase = {}
    tables = []  # (node, parents, entries, token)

    def lookup(name):
        if name in states:
            return name
        matches = lowercase.get(name.lower(), [])
        return matches[0] if len(matches) == 1 else None

    def resolve(name, token):
        if (result := lookup(name)) is None:
            raise parser.error(f"Undeclared variable {name!r}", token)
        return result

    while (token := parser.peek()) is not None:
        parser.pos += 1
        if token.is_word("network"):
            parser.name("{", "a network name")
            _skip_block(parser)
        elif token.is_word("variable"):
            node = parser.name("{", "a variable name")
            if node in states:
                raise parser.error(f"Variable {node!r} is declared twice", token)
            states[node] = _variable_block(parser, node)
            lowercase.setdefault(node.lower(), []).append(node)
        elif token.is_word("probability"):
            parser.expect("(")
            head = parser.peek()
            first = parser.words("|,)")
            if not first:
                raise parser.error("Expected a variable name", head)
            if parser.accept("|") or parser.accept(","):
                node = " ".join(w.text for w in first)
                parents = parser.names(")", "a parent name", split_words=False)
                if not parents:
                    raise parser.error("Expected a parent name")
            else:
                # Either one (multi-word) name or the old format
                # "probability ( node parent1 parent2 )"; decided once
                # every variable is declared.
                node, parents = [w.text for w in first], None
            parser.expect(")")
            tables.append((node, parents, _probability_block(parser), head))
        else:
            raise parser.error(
                "Expected 'network', 'variable' or 'probability', "
                f"got {token.text!r}",
                token,
            )

    if not states:
        raise ModelFileError("The BIF data declares no variables")

    cpds = []
    seen = set()
    for node, parents, entries, token in tables:
        if parents is None:
            words = node
            if len(words) > 1 and lookup(" ".join(words)) is not None:
                node, parents = " ".join(words), []
            else:
                node, parents = words[0], words[1:]
        node = resolve(node, token)
        parents = [resolve(p, token) for p in parents]
        if node in seen:
            raise parser.error(f"Variable {node!r} has two probability blocks", token)
        seen.add(node)
        cpds.append(_cpd(node, parents, entries, states, parser, token))

    missing = [node for node in states if node not in seen]
    if missing:
        raise ModelFileError(f"No probability block for the variables {missing}")

    return DiscreteBayesianNetwork(states=states, cpds=cpds)


def _skip_block(parser):
    """Skip a ``{ property ...; ... }`` block."""
    parser.expect("{")
    while not parser.accept("}"):
        token = parser.next("'}'")
        if token.is_word("property"):
            parser.skip_statement()
        else:
            raise parser.error(f"Unexpected {token.text!r}", token)


def _variable_block(parser, node):
    parser.expect("{")
    result = None
    while not parser.accept("}"):
        token = parser.next("'}'")
        if token.is_word("property"):
            parser.skip_statement()
            continue
        if not token.is_word("type"):
            raise parser.error(f"Unexpected {token.text!r}", token)
        kind = parser.next("a variable type")
        if not kind.is_word("discrete"):
            raise NotImplementedError(
                f"BIF variable {node!r} has type {kind.text!r} on line "
                f"{kind.line}; only discrete variables are supported"
            )
        parser.expect("[")
        size = parser.next("the number of states")
        try:
            num_states = int(size.text)
        except ValueError:
            raise parser.error(
                f"Expected the number of states, got {size.text!r}", size
            ) from None
        parser.expect("]")
        parser.expect("{")
        values = parser.names("}", "a state")
        parser.expect("}")
        parser.expect(";")
        if len(values) != num_states:
            raise parser.error(
                f"Variable {node!r} declares {num_states} states but lists "
                f"{len(values)}",
                size,
            )
        if len(set(values)) != len(values):
            raise parser.error(f"Variable {node!r} has repeated states", size)
        result = values
    if result is None:
        raise parser.error(f"Variable {node!r} has no type")
    return result


def _probability_block(parser):
    """The entries of a probability block: ``(kind, key, values, token)``."""
    parser.expect("{")
    entries = []
    while not parser.accept("}"):
        token = parser.next("'}'")
        if token.is_word("property"):
            parser.skip_statement()
        elif token.is_word("table") or token.is_word("default"):
            entries.append((token.text, None, parser.numbers(), token))
        elif token.is_punct("("):
            key = parser.names(")", "a parent state", split_words=False)
            parser.expect(")")
            entries.append(("row", tuple(key), parser.numbers(), token))
        else:
            raise parser.error(f"Unexpected {token.text!r}", token)
    return entries


def _cpd(node, parents, entries, states, parser, token):
    node_states = states[node]
    parent_states = [states[p] for p in parents]
    num_configs = 1
    for s in parent_states:
        num_configs *= len(s)

    rows = {}
    default = None
    for kind, key, values, entry_token in entries:
        if kind == "table":
            if rows or default is not None or len(entries) > 1:
                raise parser.error(
                    f"The probability block of {node!r} mixes 'table' with "
                    "other entries",
                    entry_token,
                )
            expected = len(node_states) * num_configs
            if len(values) != expected:
                raise parser.error(
                    f"The table of {node!r} has {len(values)} values, "
                    f"expected {expected}",
                    entry_token,
                )
            # The node's state varies slowest: values[i * num_configs + j].
            return _make_cpd(
                node,
                parents,
                parent_states,
                lambda j: values[j::num_configs],
            )

        if len(values) != len(node_states):
            raise parser.error(
                f"An entry of {node!r} has {len(values)} values, expected "
                f"{len(node_states)}",
                entry_token,
            )
        if kind == "default":
            if default is not None:
                raise parser.error(
                    f"The probability block of {node!r} has two defaults",
                    entry_token,
                )
            default = values
            continue

        if len(key) != len(parents):
            raise parser.error(
                f"An entry of {node!r} has {len(key)} parent states, expected "
                f"{len(parents)}",
                entry_token,
            )
        for state, p, ps in zip(key, parents, parent_states):
            if state not in ps:
                raise parser.error(
                    f"Unknown state {state!r} of {p!r} in an entry of {node!r}",
                    entry_token,
                )
        if key in rows:
            raise parser.error(
                f"The probability block of {node!r} repeats the parent states "
                f"{key}",
                entry_token,
            )
        rows[key] = values

    if not parents:
        if () not in rows and default is None:
            raise parser.error(f"The probability block of {node!r} is empty", token)
        return _make_cpd(node, parents, parent_states, lambda j: rows.get((), default))

    keys = _parent_configs(parent_states)
    missing = [key for key in keys if key not in rows]
    if missing and default is None:
        raise parser.error(
            f"The probability block of {node!r} has no entry for the parent "
            f"states {missing[0]}",
            token,
        )
    return _make_cpd(node, parents, parent_states, lambda j: rows.get(keys[j], default))


def _parent_configs(parent_states):
    """Every parent configuration, the last parent varying fastest."""
    configs = [()]
    for s in parent_states:
        configs = [c + (v,) for c in configs for v in s]
    return configs


def _make_cpd(node, parents, parent_states, distribution):
    """A CPD whose distribution for the ``j``-th parent configuration is
    ``distribution(j)``."""
    if not parents:
        return DiscreteCPD(node=node, values=list(distribution(0)))
    keys = _parent_configs(parent_states)
    if len(parents) == 1:
        keys = [key[0] for key in keys]
    return DiscreteCPD(
        node=node,
        parents=parents,
        values={key: list(distribution(j)) for j, key in enumerate(keys)},
    )


def save_bif(pgm, filename=None, name="unknown"):
    """Write a Bayesian network in BIF format.

    Node and state names are written with ``str`` (and quoted unless they
    consist of letters, digits and ``_.+-``), so a network loaded back has
    string nodes and states. The probabilities are written with ``repr``,
    so they are read back exactly.

    Parameters
    ----------
    pgm : DiscreteBayesianNetwork
        The network to write.
    filename : str, optional
        Output file name. If omitted, the BIF text is returned.
    name : str, optional
        The network name written in the file.

    Returns
    -------
    str or None
        The BIF text if ``filename`` is omitted, else None.

    Raises
    ------
    TypeError
        If ``pgm`` is not a ``DiscreteBayesianNetwork``.
    InvalidInputError
        If two nodes, or two states of a node, have the same string form, or
        a name contains a double quote.
    """
    if not isinstance(pgm, DiscreteBayesianNetwork):
        raise TypeError(
            "The BIF format holds a DiscreteBayesianNetwork, got "
            f"{type(pgm).__name__}"
        )
    text = _bif_text(pgm, name)
    if filename is None:
        return text
    with open(filename, "w", encoding="utf-8") as OUTPUT:
        OUTPUT.write(text)


def _quote(label):
    text = str(label)
    if '"' in text:
        raise InvalidInputError(f"Cannot write the name {text!r} in BIF: it has a '\"'")
    if _PLAIN_RE.fullmatch(text) and text not in _KEYWORDS:
        return text
    return f'"{text}"'


def _check_unique(labels, what):
    strings = [str(label) for label in labels]
    if len(set(strings)) != len(strings):
        raise InvalidInputError(
            f"Cannot write {what} {list(labels)} in BIF: their string forms repeat"
        )


def _bif_text(pgm, name):
    nodes = pgm.nodes
    _check_unique(nodes, "the nodes")
    lines = [f"network {_quote(name)} {{", "}"]
    for node in nodes:
        states = pgm.states_of(node)
        _check_unique(states, f"the states of {node!r}")
        lines += [
            f"variable {_quote(node)} {{",
            f"  type discrete [ {len(states)} ] "
            f"{{ {', '.join(_quote(s) for s in states)} }};",
            "}",
        ]

    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    for node in nodes:
        cpd = cpds[node]
        states = pgm.states_of(node)
        parents = list(cpd.parents or [])

        def probabilities(dist):
            return ", ".join(repr(float(dist[s])) for s in states)

        if not parents:
            lines += [
                f"probability ( {_quote(node)} ) {{",
                f"  table {probabilities(cpd.values)};",
                "}",
            ]
            continue
        lines.append(
            f"probability ( {_quote(node)} | "
            f"{', '.join(_quote(p) for p in parents)} ) {{"
        )
        for key in _parent_configs([pgm.states_of(p) for p in parents]):
            dist = cpd.values[key[0] if len(parents) == 1 else key]
            lines.append(
                f"  ({', '.join(_quote(s) for s in key)}) {probabilities(dist)};"
            )
        lines.append("}")
    return "\n".join(lines) + "\n"

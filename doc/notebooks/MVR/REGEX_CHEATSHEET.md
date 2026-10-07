# MVR regex syntax — cheatsheet

`mvr_regex` compiles a regular expression over hidden-state labels into a
`HomMVR`. The pattern becomes a minimal DFA and that DFA's states become the
mediation states, so a pattern is not a new kind of constraint — it is a compact
way to write an automaton by hand. Worked demo:
[`regex_tutorial.ipynb`](regex_tutorial.ipynb).

```python
from conin.hidden_markov_model.mvr_constraints import mvr_regex

mvr_regex(hmm, "<a><b>*<c>")            # -> HomMVR
build_mvr(hmm, 'match "<a><b>*<c>"')    # the same automaton, from the formula layer
```

## 1. Dependencies
This is the one constructor with an optional dependency: it needs
[`greenery`](https://pypi.org/project/greenery/) (`pip install conin[regex]`),
and without it the call raises `ImportError`.

## 2. Hidden State Labels

A hidden state is always written `<label>`, using the label's `str()` form.
Everything outside the angle brackets is regular-expression syntax. A bare
character is an **error rather than a literal**, so a pattern can never quietly
mean something other than what it reads as.

## 3. Syntax

Examples show which of `a`, `b`, `c`, `aa`, `ab`, `ac`, `bc`, `cc`, `abb`, `aaa`,
`abc` the pattern accepts, over states `a b c`.

| Syntax | Means | Example | Accepts |
| --- | --- | --- | --- |
| `<label>` | that hidden state | `<a>` | `a` |
| `.` | any hidden state | `<a>.` | `aa`, `ab`, `ac` |
| `\|` | alternation | `<a>\|<b><c>` | `a`, `bc` |
| `( … )` | grouping | `(<a><b>)*<c>` | `c`, `abc` |
| `*` | zero or more | `<a><b>*` | `a`, `ab`, `abb` |
| `+` | one or more | `<a><b>+` | `ab`, `abb` |
| `?` | optional | `<a><b>?` | `a`, `ab` |
| `{m}`, `{m,n}`, `{m,}` | bounded repetition | `<a>{2,3}` | `aa`, `aaa` |
| `[ … ]` | one of a set of states | `[<a><b>]<c>` | `ac`, `bc` |
| `[^ … ]` | any state except | `[^<a>]<c>` | `bc`, `cc` |
| `[<a>-<b>]` | a range of states | `[<a>-<b>]<c>` | `ac`, `bc` |

## 4. A pattern matches the whole sequence

There is no substring match and no anchors. The MVR evaluates to `True` at time
`t` exactly when full hidden sequence (over its `time_rnage`) matches the pattern. This has three consequences:

 - **Nothing needs to be anchored.** `^` or `$` is rejected with its own
   error message saying so.
 - **A time range pins the length.** Under a `time_range` of `[a, b]` the pattern
   must match `hidden[a .. b]`, which is a segment of fixed length. So
   `<a><b>*` — three sequences on its own — accepts only `abb` over the window
   `[1, 3]`.
 - **The empty string matches nothing**, since an MVR
   always consumes at least one hidden state. While `""` and `<a>{0}` are built anyway
   as constantly-false constraints, this will raise a `UserWarning`.

## 5. Ranges follow the model's state order

`[<a>-<c>]` is a range over the positions of `hmm.hidden_states`, not over the
text of the labels. For example, with hidden states `1`, `2`, `10`:

 - `[<1>-<2>]` accepts `1` and `2` — the first two positions.
 - `[<1>-<10>]` accepts `1`, `2` **and** `10` — every position from the first to
   the last, even though `10` sorts before `2` as text.

## 6. Invalid Syntax

Each of these is a construction-time error naming the offending index, so a
malformed pattern fails immediately rather than matching something unintended.

| Pattern | Message |
| --- | --- |
| `<a>b` | `'b' at index 3 of pattern is not regex syntax here; write every hidden state as <label>` |
| `^<a>` | `'^' at index 0 of pattern is an anchor, which is neither supported nor needed: a pattern always matches the full sequence` |
| `<a` | `unterminated < at index 0 of pattern` |
| `<z>` | `<z> is not a hidden state of hidden_markov_model` |
| `\d` | `'\\' at index 0 of pattern is not regex syntax here; write every hidden state as <label>` |

So backslash escapes, shorthand classes (`\d`, `\w`), and backreferences are all
invalid, along with bare literals. One further restriction is on state labels: a state whose label contains `<` or `>` cannot be named
by a pattern at all, because those characters delimit a label. That constraint must be build by hand or by calling constructors/operators.

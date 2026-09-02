# MVR formula syntax — cheatsheet

A formula string is parsed into calls to `mvr_constraints.py` and
`mvr_operators.py`. Nothing new is constructed, so every formula below is
shorthand for calls you could write by hand. Worked demo: [`formula.ipynb`](formula.ipynb).

```python
from conin.hidden_markov_model.mvr_formula import build_mvr, build_mvr_functor

build_mvr(hmm, "reach B before A[2]")     # -> list[BaseMVR],       for MVR_CHMM(constraints=...)
build_mvr_functor("reach B before A[2]")  # -> list[MVRConstraint], for ConstrainedHiddenMarkovModel(constraints=...)
```

Both return a **list**, one entry per top-level `and` (more efficient to provide a list of active constraints than build a single big constraint). `build_mvr` requires an `hmm` object and builds a list of MVRs for that HMM. `build_mvr_functor` builds a list of `MVRFunctor`, factories that create `mvr` objects for a given `hmm`.

Each result is named after its own formula — `"never A and reach B[0]"` — unless
you pass `name=`. That name is what `sat_time_mvr` and `sat_prob_mvr` accept as
`target="..."`.

## Atoms

Examples show which of `a`, `ab`, `aab`, `aba` the formula accepts, over states `a b c`.

| Syntax | Means | Lowers to |
| --- | --- | --- |
| `A` | current state is `A` | `mvr_current_state` |
| `(A, B)` | current state is `A` or `B` | `mvr_current_state` |
| `<A>` | same as `A`; protection if states contain special characters | `mvr_current_state` |
| `(<A>, <B>)` | an escaped state set; escaped and bare labels can be mixed | `mvr_current_state` |
| `true` / `false` | constantly true / false | `mvr_constant` |
| `A -> B` | the last transition was `A` to `B` | `mvr_current_transition` |
| `seq(A, B, C)` | the path ends with `A B C`, adjacent | `mvr_current_sequencelist` |
| `jump` | current state differs from the previous one | `mvr_jump` |
| `hold k` | every run has length `>= k` (trailing run exempt) | `mvr_holdingtime` |
| `hold k in (A, B)` | as above, but only for runs of `A` or `B` | `mvr_holdingtime` |
| `jumps COND` | the number of jumps satisfies `COND` | `mvr_jumpcounts` |
| `within [lo, hi]` | current state is in `[lo, hi]` — **numeric states only** | `mvr_withinbounds` |
| `within {t: [lo, hi], ...}` | per-time bounds; unnamed times unconstrained | `mvr_withinbounds` |
| `match "<a><b>*"` | the path matches the regex — **needs `greenery`** | `mvr_regex` |
| `(E)` | grouping | |

**Escaping a label.** A bare label must look like a Python identifier
(`A`, `wash_2`). Write it as `<label>` whenever it does not — because it is a
keyword (`<count>`, `<then>`), a number (`<3>`), or contains punctuation
(`<A-1>`, `<B.2>`, `<C$>`, `<pkt/sec>`). The escape works anywhere a label is
accepted: in sets, `seq(...)`, `hold k in (...)` and `A -> B`.

Three characters cannot appear in a label named by a formula: **whitespace**,
**`<`** and **`>`**. Each is a syntax error rather than a silent misreading, but
such a state is unreachable from the formula layer — build that constraint by
calling the constructors directly. (`mvr_regex` rejects `<` and `>` in labels for
the same reason.)

## Operators

| Syntax | Means | Lowers to |
| --- | --- | --- |
| `not E` | `E` is false **right now** | `mvr_not` |
| `never E` | `E` has never held, at any time up to now | `mvr_not_yet` |
| `reach E` | `E` has held at some time up to now | `mvr_already_satisfied` |
| `first E` | `E`, made prefix-free: accepts only the first time it holds | `mvr_sattime` |
| `reverse E` | the reversed language of `E` | `mvr_reverse` |
| `E and F` | both | list, or `mvr_and` when nested |
| `E or F` | either | `mvr_or` |
| `E but not F` | `E` and not `F`, as languages | `mvr_setdiff` |
| `E cat F` | language concatenation | `mvr_concatenate` |
| `E{k}` | `E` repeated `k` times | `mvr_kfold_product` |
| `E+` | `E` repeated one or more times | `mvr_kleene_closure` |
| `E[k]` | `E` has held exactly `k` times — sugar for `count(E) k` | `mvr_count` |
| `count(E) COND` | the number of times `E` has held satisfies `COND` | `mvr_count` |
| `E then F`, `E before F` | `E` first holds strictly before `F` does | `mvr_precedence` `<` |
| `E after F` | `E` first holds strictly after `F` | `mvr_precedence` `>` |
| `E at or before F` | | `mvr_precedence` `<=` |
| `E at or after F` | | `mvr_precedence` `>=` |
| `E between a and b` | enforce `E` only over times `a..b` | `mvr_timerange` |
| `E over [a, b]` | same thing | `mvr_timerange` |

Neither operand of a temporal operator has to hold: `E before F` is satisfied as
soon as `E` holds while `F` still does not.

## Count conditions

Used by `count(E) ...` and `jumps ...`. This is `mvr_count`'s own mini-language,
passed through unchanged.

| Syntax | Means |
| --- | --- |
| `2`, `== 2`, `is 2` | exactly 2 |
| `>= 2`, `> 2`, `<= 2`, `< 2` | inequality |
| `in [1,3]` | between 1 and 3 inclusive |
| `in (1,3]` | 2 or 3 |

`< 0` and `>= 0` are rejected as degenerate.

## Operator Priority

Highest priority first — the top of the list binds tightest and claims its
operands first. Parenthesise to override.

```
{k} · + · [k]                                (postfix)
not · never · reach · first · reverse        (prefix)
cat
then · before · after · at or before · at or after
between / over
but not
and
or
```

The three set operators — `but not`, `and`, `or` (difference, intersection,
union) — cluster at the bottom, below `cat`. That follows Ragel, which groups
`-` with `&` and `|` beneath concatenation, and SVA, which puts `intersect`,
`and` and `or` beneath `##`.

## Common Pitfalls

- **`not A` is not `never A`.** `not A` is about this instant; `never A` is about
  all of history. Over states `a b c`:

  | formula | `a` | `b` | `ab` | `aab` | `aba` |
  | --- | --- | --- | --- | --- | --- |
  | `not a` | ✗ | ✓ | ✓ | ✓ | ✗ |
  | `never a` | ✗ | ✓ | ✗ | ✗ | ✗ |

- **A bare label means "currently in that state"**, not "visits that state". `a`
  accepts `a`, `ba`, `aa`, `aba` — every path *ending* in `a`. Wrap it in `reach`
  for "visits at some point". This coercion is why `never` and `reach` work on any
  expression rather than only on state names.

- **`then` is precedence, not concatenation.** Temporal words all name
  `mvr_precedence`; the regular-language concatenation operator is `cat`. `a cat b`
  accepts `ab`, `aab`, `bab`, `abab` — an `a` somewhere, then a `b`, ending at the
  `b`.

- **Temporal operators chain like Python comparisons.** `reach A then reach B then
  reach C` means `(A then B) and (B then C)`, so it splits into two constraints. A
  chain must use one relation throughout; mixing them is rejected rather than
  guessed at.

- **A top-level `and` splits into separate constraints** rather than building the
  product automaton — downstream algorithms take a list of MVRs and are cheaper for
  it. A **nested** `and` does build the product.

- **`between` binds tighter than `and`.** `never A and reach B between 2 and 5`
  windows only `reach B`. Write `(never A and reach B) between 2 and 5` to window
  both; the window is then pushed onto each conjunct.

- **A `time_range` is where the constraint is *enforced*, not a restart.** The
  automaton is initialized at `a` and evaluated at `b`, and contributes nothing
  outside `[a, b]` — it never sees the states before `a`. So a windowed constraint
  is not the same as slicing the result of an unwindowed one.

- **`E[k]` is exactly `k`, not at least `k`.** `a[2]` accepts `aa` but not `aaa`.
  It reads as "the k-th occurrence" under a temporal operator because precedence
  compares *first* satisfaction times, and `count == k` first holds at the k-th
  occurrence — which is what makes `reach B before A[2]` mean "B before the 2nd A".

- **Labels resolve by `str()`**, so numeric hidden states work. A label that
  collides with a keyword, is a number, or contains punctuation must be written
  `<label>`; see **Escaping a label** above for the three characters that cannot be
  escaped at all.

## Examples

| Formula | Reading |
| --- | --- |
| `never A` | never visit `A` |
| `reach B` | visit `B` at some point |
| `reach B before A[2]` | hit `B` before the 2nd visit to `A` |
| `count(B) in [2,4]` | between two and four `B`s |
| `never A between 0 and 4` | avoid `A` over the first five steps only |
| `never A and reach B and reach C` | three separate constraints |
| `seq(A, B) then reach C` | an adjacent `A B` occurs before `C` is reached |
| `hold 3 in (A, B)` | runs of `A` and of `B` last at least 3 steps |
| `jumps <= 2` | at most two state changes |
| `A -> B` | the path ends on the transition `A` to `B` |
| `never (A -> B)` | that transition never occurs |
| `match "<A><B>*"` | one `A` followed by any number of `B`s |
| `reverse (A cat B)` | a `B` before an `A`, read backwards |

Anything the grammar cannot say is still reachable by calling `mvr_constraints`
and `mvr_operators` directly. The formula layer is a convenience over that
algebra, not a replacement for it.

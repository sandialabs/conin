# MVR formula syntax — cheatsheet

A formula string is parsed into calls to `mvr_constraints.py` and
`mvr_operators.py`. Nothing new is constructed, so every formula below is
shorthand for calls you could write by hand. Worked demo: [`formula.ipynb`](formula.ipynb).

```python
from conin.hidden_markov_model.mvr_formula import build_mvr, build_mvr_functor

build_mvr(hmm, "reach B before A[2]")     # -> list[BaseMVR],       for MVR_CHMM(constraints=...)
build_mvr_functor("reach B before A[2]")  # -> list[MVRConstraint], for ConstrainedHiddenMarkovModel(constraints=...)
```

Both return a **list**, one entry per top-level `and`. Downstream algorithms take
a list of active constraints, and that is cheaper than building one big combined
constraint. The two differ only in when the MVR is built:

 - `build_mvr` takes an `hmm` and builds the MVRs for that HMM right away.
 - `build_mvr_functor` takes no `hmm`. It builds `MVRConstraint` factories, each of
   which constructs its MVR later, when `initialize_chmm` supplies an HMM.

Each result is named after its own formula — `"never A and reach B[0]"` — unless
you pass `name=`. That name is what `sat_time_mvr` and `sat_prob_mvr` accept as
`target="..."`.

## 1. Atoms

Examples show which of `a`, `ab`, `aab`, `aba` the formula accepts, over states `a b c`.

| Syntax | Means | Lowers to |
| --- | --- | --- |
| `A` | current state is `A` | `mvr_current_state` |
| `(A, B)` | current state is `A` or `B` | `mvr_current_state` |
| `<A>` | escape; same as `A` | `mvr_current_state` |
| `(<A>, <B>)` | escaped state set | `mvr_current_state` |
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

### 1a. Escaping a label 
Sometimes, a state label might contain reserved characters in the syntax. For example, it might be a
keyword (`<count>`, `<then>`), a number (`<3>`), or contain punctuation
(`<A-1>`, `<B.2>`, `<C$>`, `<pkt/sec>`). Use `< >` as an escape in those cases. This works anywhere a label is
accepted: in sets, `seq(...)`, `hold k in (...)` and `A -> B`. However, three characters cannot appear in a label named by a formula: 

 - whitespace
 - `<`
 - `>`

Having one of the three in a state label will raise syntax error. Build that constraint by
calling the constructors directly. (`mvr_regex` rejects `<` and `>` in labels for
the same reason.)

### 1b. Regex patterns

`match "..."` hands its pattern straight to `mvr_regex`, which has a syntax of its
own. Hidden states are written `<label>` there too, but for a different reason: in
a pattern every bare character is regex syntax, so a state must always be bracketed
rather than only when it would be ambiguous. See
[`REGEX_CHEATSHEET.md`](REGEX_CHEATSHEET.md).

## 2. Operators

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
| `E[k]` | `E` has held exactly `k` times — same as `count(E) k` | `mvr_count` |
| `count(E) COND` | the number of times `E` has held satisfies `COND` | `mvr_count` |
| `E then F`, `E before F` | `E` first holds strictly before `F` does | `mvr_precedence` `<` |
| `E after F` | `E` first holds strictly after `F` | `mvr_precedence` `>` |
| `E at or before F` | | `mvr_precedence` `<=` |
| `E at or after F` | | `mvr_precedence` `>=` |
| `E between a and b` | enforce `E` only over times `a..b` | `mvr_timerange` |
| `E over [a, b]` | same thing | `mvr_timerange` |

### Temporal Semantics
Neither operand of a temporal operator has to hold: `E before F` is satisfied as
soon as `E` holds while `F` still does not. It merely enforces that the temporal order must hold. If one wants to enforce both `E` and `F` satisfied, with `E` satisfied first, then `(E before F) and F` suffices.

### 2b. Count conditions

Used by `count(E) ...` and `jumps ...`. This is `mvr_count`'s own mini-language,
passed through unchanged.

| Syntax | Means |
| --- | --- |
| `2`, `== 2`, `is 2` | exactly 2 |
| `>= 2`, `> 2`, `<= 2`, `< 2` | inequality |
| `in [1,3]` | between 1 and 3 inclusive |
| `in (1,3]` | 2 or 3 |

`< 0` and `>= 0` are rejected as degenerate.

## 3. Operator Priority

High priority listed first. Parentheses to override.

1. `{k}`, `+`, `[k]` (postfix)
2. `not`, `never`, `reach`, `first`, `reverse` (prefix)
3. `cat`
4. `then`, `before`, `after`, `at or before`, `at or after` (temporal)
5. `between`, `over`
6. `but not`
7. `and`
8. `or`

## 4. Common Pitfalls

- **`not A` is not `never A`.** `not A` is about this instant; `never A` is about
  all of history. Over states `a b c`:

  | formula | `a` | `b` | `ab` | `aab` | `aba` |
  | --- | --- | --- | --- | --- | --- |
  | `not a` | ✗ | ✓ | ✓ | ✓ | ✗ |
  | `never a` | ✗ | ✓ | ✗ | ✗ | ✗ |

- **A label means "currently in that state"**, not "visits that state". `a`
  accepts `a`, `ba`, `aa`, `aba` — every path *ending* in `a`. Wrap it in `reach`
  for "visits at some point". This is why `never` and `reach` work on any
  expression.

- **`then` is precedence, not concatenation.** Temporal words all name
  `mvr_precedence`; the concatenation operator is `cat`: eg. `a cat b`.

- **Temporal operators chain like Python comparisons.** `reach A then reach B then
  reach C` means `(A then B) and (B then C)`, so it splits into two constraints. A
  chain must use one relation throughout; mixing them is rejected rather than
  guessed at.

- **A top-level `and` splits into separate constraints** rather than building the
  product automaton — downstream algorithms take a list of MVRs and are cheaper for
  it. A **nested** `and` does build the product.

- **`between` has higher priority than `and`.** `never A and reach B between 2 and 5`
  windows only `reach B`. Write `(never A and reach B) between 2 and 5` to window
  both.

- **A `time_range` is where the constraint is *enforced*, not a restart.** The
  automaton is initialized at `a` and evaluated at `b`, and contributes nothing
  outside `[a, b]`.

- **`E[k]` is exactly `k`, not at least `k`.** `a[2]` accepts `aa` but not `aaa`.

- **Labels resolve by `str()`**, so numeric hidden states work. A label that
  collides with a keyword, is a number, or contains punctuation must be written
  `<label>`; see **Escaping a label** above for the three characters that cannot be
  escaped at all.

## 5. Examples

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

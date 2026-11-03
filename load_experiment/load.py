"""Memory load of a multi-step calculation.

A step's result is "live" from the moment it is computed until its last use.
The load of a problem is the smallest possible peak number of live results,
taken over every valid order of the steps. Numbers that are written in the
question text never count, because the model can always read them back.

Steps are described by `uses`: uses[i] is the set of step indices whose
results step i reads. Example (both have 3 steps):

    chain  4*6=24 -> 24-5=19 -> 19*2=38   uses = [set(), {0}, {1}]     load 1
    wide   4*6=24,  3*5=15  -> 24-15=9    uses = [set(), set(), {0, 1}] load 2
"""
from functools import lru_cache

MAX_STEPS = 20  # the search below visits up to 2**n subsets


def _users(uses):
    n = len(uses)
    return [{w for w in range(n) if u in uses[w]} for u in range(n)]


def _check(uses):
    n = len(uses)
    if n > MAX_STEPS:
        raise ValueError(f"{n} steps is too many for the exact search (max {MAX_STEPS})")
    for i, deps in enumerate(uses):
        for d in deps:
            if not 0 <= d < n or d == i:
                raise ValueError(f"step {i} reads invalid step {d}")


def live_count(done, uses, users=None):
    """Number of computed results (in `done`) that a step not yet computed still needs."""
    users = users if users is not None else _users(uses)
    return sum(1 for u in done if any(w not in done for w in users[u]))


def order_load(order, uses):
    """Peak live count when the steps are computed in the given order."""
    _check(uses)
    if sorted(order) != list(range(len(uses))):
        raise ValueError("order must list every step exactly once")
    users = _users(uses)
    done, peak = set(), 0
    for v in order:
        if not uses[v] <= done:
            raise ValueError(f"step {v} runs before a step it reads")
        peak = max(peak, live_count(done, uses, users))
        done.add(v)
    return peak


def min_load(uses):
    """Lowest possible peak live count over all valid orders of the steps."""
    _check(uses)
    n = len(uses)
    if n == 0:
        return 0
    users = _users(uses)
    full = (1 << n) - 1

    def live(mask):
        return sum(1 for u in range(n)
                   if mask >> u & 1 and any(not mask >> w & 1 for w in users[u]))

    @lru_cache(maxsize=None)
    def best(mask):
        if mask == full:
            return 0
        ready = [v for v in range(n)
                 if not mask >> v & 1 and all(mask >> p & 1 for p in uses[v])]
        if not ready:
            raise ValueError("steps contain a cycle")
        return max(live(mask), min(best(mask | 1 << v) for v in ready))

    return best(0)

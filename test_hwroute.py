"""Run: python test_hwroute.py"""
from collect import grade
from router import choose, routellm_quality

# Two backends, weak then strong: beta=0 is RouteLLM's threshold rule.
cost, latency = [0.2, 1.0], [1.0, 0.6]
for alpha in (0.1, 0.5, 1.0):
    for p in (i / 100 for i in range(101)):
        if abs(p - alpha * 0.8) > 1e-9:  # skip the exact tie
            assert (choose(routellm_quality(p, [0, 1]), cost, latency, alpha, 0) == 1) == (p > alpha * 0.8)

# A faster strong backend wins even an easy query once latency matters enough.
assert choose(routellm_quality(0.0, [0, 1]), cost, latency, 0.1, 0) == 0
assert choose(routellm_quality(0.0, [0, 1]), cost, latency, 0.1, 1) == 1

# Same model on two chips: cheaper one when only cost matters, faster one when latency does.
q = routellm_quality(0.9, [0, 1, 1])
assert choose(q, [0.2, 1.0, 1.5], [1.0, 0.6, 0.1], 0.5, 0) == 1
assert choose(q, [0.2, 1.0, 1.5], [1.0, 0.6, 0.1], 0.5, 2) == 2

# A middle model takes the middling queries: easy ones go down, hard ones go up.
picks = choose(routellm_quality([0.1, 0.5, 0.95], [0, 0.6, 1]), [0.2, 0.5, 1.0], [1.0, 1.0, 0.6], 0.4, 0)
assert list(picks) == [0, 1, 2]

assert grade("Answer: B", "B") and grade("Some reasoning.\nAnswer: **C**", "C") and grade("J", "J")
assert grade("The answer is (D).", "D") and not grade("Answer: B", "C") and not grade("", "A")
assert not grade("I think it is B", "I")  # the pronoun is not an answer
print("ok")

# Notes — ch01 studies

## Study A: independent Q-learners on matching pennies

**Measured result (seed 0, 30k steps):** mean radial deviation of the
`(p0_heads, p1_heads)` trajectory from (0.5, 0.5) is **0.125** over the
full trace, with p0 range roughly (0.17, 0.83). The two agents' P(heads)
traces oscillate persistently and neither converges — the "closed cycle"
of the deterministic replicator idealisation shows up empirically as a
noisy disk around the mixed equilibrium.

**Hyperparameter provenance.** Learning rate 0.15 and softmax temperature
1.0 were chosen from a 16-point sweep over (lr ∈ {0.005, 0.01, 0.02, 0.05,
0.1, 0.15, 0.2, 0.3}) × (tau ∈ {1, 2, 3, 5}). At small lr with high tau,
Q values dampen fast and the agents damp toward (0.5, 0.5) — no orbit.
At small tau, the softmax saturates and the trace pins to the corners
(1, 0) → (0, 1) → ..., which looks like a square-wave, not an orbit. The
chosen point (lr=0.15, tau=1.0) is the smallest-radius setting that
maintains a **visibly sustained** orbit for the full 30k steps; smaller
radii require larger tau and start looking like damping toward the fixed
point.

**A deterministic ablation** (Q updates use the exact expected reward
against the opponent's *current* softmax, no episode-sampling noise)
converges cleanly to (0.5, 0.5) with any positive lr under 1. That the
stochastic version *doesn't* is the point of the study — the noise a
learning agent has to deal with is what breaks the fixed-point argument.

## Study B: joint-policy correlation on a 5-colour matching game

**Measured result (seed 0):** cross-play matrix diagonal mean = **1.00**,
off-diagonal mean = **0.10**, **gap = 0.90**. The five pairs each landed
on a distinct colour (they're seeded 0, 10, 20, 30, 40, which happen to
select colours (0, 1, 2, 3, 4) in that order at these hyperparameters).
Cross-play between agents from different pairs collides only by chance
(1/k = 0.20), and after 20k training steps each pair's colour choice is
sharp enough that even chance collisions are rare — 0.10 rather than 0.20.

If a run's pairs happen to duplicate colours (which can occur with
different seed schedules), the off-diagonal mean rises. Rerunning with a
few different `--seed` values gave gaps in the 0.7–0.9 range; the seed-0
number above is representative of the chapter's "strong diagonal" claim.

**The number to quote in §1.3** is that 0.90 gap (or the sentence "the
diagonal averages 1.00 and the off-diagonal 0.10 — a 5× reduction in
expected reward when partners haven't trained together").

## Runtime

Both studies complete in well under a minute on the reference server (30k
steps of stateless updates + 5 × 20k paired training + 25-cell evaluation).
`--smoke` shortens to a few seconds while exercising both code paths.

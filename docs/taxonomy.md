# PSRO variant taxonomy

Transcribed verbatim from the manuscript, Chapter 9 §9.1 (Table 9.1). This
file is a **living document**: whenever a new variant is added to the repo,
add a row here so a reader can see at a glance which component the variant
changes and where the code lives. Same convention as the printed table:

- **●** = the variant's substantive change.
- **○** = incidental / inherited.
- **Loop** = the control flow of `run_psro` itself, not its components.
- **In repo** = ✔ implemented in this repository / ✘ deliberately not
  implemented (see §9.3 "taken on faith" or the per-row caveat).
- **Repo location** = the module(s) that carry the variant, added here to
  save readers a `find` — this column is the value `docs/taxonomy.md` adds
  over the printed table.

## Table 9.1 — Published variants by component changed

| Family / variant | Oracle | MSS | Evaluator | Loop | In repo | Repo location |
|---|---|---|---|---|---|---|
| Double oracle (McMahan et al. 2003) | ● exact | ● Nash | ○ exact | — | ✔ Ch. 4 | `experiments/ch04_double_oracle/`, `src/psrolab/meta_solvers/support_enum.py` |
| Fictitious play / FSP (Brown 1951; Heinrich et al. 2015) | ○ | ● uniform | ○ | — | ✔ Ch. 3/5 | `src/psrolab/baselines/fp.py`, `experiments/ch03_fictitious_play/` |
| PSRO, original (Lanctot et al. 2017) | ● RL | ● PRD | ● sampled | — | ✔ Ch. 6–8 | `src/psrolab/psro.py`, `src/psrolab/oracles/ppo.py`, `src/psrolab/meta_solvers/projected_replicator.py` |
| PSRO-Nash ("vanilla") | ○ RL | ● Nash | ○ sampled | — | ✔ Ch. 7 | `src/psrolab/meta_solvers/projection.py`, `experiments/ch07_kuhn_leduc/` |
| α-Rank PSRO (Muller et al. 2020) | ○ RL | ● α-Rank | ○ | — | ✔ Ch. 8 | `src/psrolab/meta_solvers/alpha_rank.py` |
| Regret-matching / CCE meta-solvers | ○ RL | ● RM→CCE | ○ | — | ✔ Ch. 8 | `src/psrolab/meta_solvers/regret_matching.py` |
| JPSRO / correlated-device variants | ○ RL | ● joint CE/CCE | ○ | ○ joint sampling | ✘ §8.3 note | — |
| Diverse PSRO (behavioral/response diversity) | ● RL + bonus | ○ Nash | ○ | — | ✔ Ch. 11 (simplified — see caveat) | `src/psrolab/oracles/diverse_ppo.py` |
| PSD-PSRO / determinant-style diversity | ● RL + bonus | ○ | ○ | — | ✘ §11.2 deltas | — |
| Self-play | ○ RL | ● point mass (last) | ○ | — | ✔ Ch. 12 | `src/psrolab/baselines/self_play.py` |
| Last-k / windowed | ○ RL | ● uniform over last k | ○ | — | ✔ Ch. 12 | `src/psrolab/baselines/last_k_solver.py` |
| Warm-started BRs (Fusion-style init) | ● RL + init | ○ | ○ | — | ✘ §12.4 | — |
| Anytime / optimistic PSRO | ○ | ● no-regret target | ○ | ○ | ✘ §12.2 | — |
| Pipeline PSRO (McAleer et al. 2020) | ○ RL ×k | ○ Nash | ○ | ● async levels | ✔ Ch. 10 (parallel — see caveat) | `src/psrolab/psro_pipeline.py`, `experiments/ch10_pipeline/` |
| Distributed evaluation | ○ | ○ | ● parallel | ○ | ✔ Ch. 10 | `src/psrolab/psro_pipeline.py` (Ray evaluator actor) |
| XDO / extensive-form DO (McAleer et al. 2021) | ○ RL | ○ | ○ | ● restricted game redefined | ✘ §12.3 | — |
| TE-PSRO / transfer-enhanced | ● RL + transfer | ○ | ○ | ○ | ✘ §12.3 | — |

## Caveats on two ✔ rows

**Pipeline PSRO.** The repo implements *synchronous parallel PSRO* —
parallelism within an iteration, using Ray actors to train k best responses
simultaneously against the same fixed opponent mixture. It is **not**
Pipeline PSRO proper, which removes the iteration barrier and maintains a
hierarchical pipeline of workers each training against lower levels. The
✔ is for the parallel loop, not for a faithful Pipeline PSRO
reproduction. See `src/psrolab/psro_pipeline.py` docstring for the
implementation delta.

**PSD-PSRO / diverse PSRO row.** `oracles/diverse_ppo.py` implements a
*simplified* behavioral-diversity bonus: per-visited-state TV distance to
the nearest population member, added to the PPO reward. This is not the
full occupancy-weighted determinantal objective of Yao et al. 2023 (which
is why the PSD-PSRO row is marked ✘ separately, deliberately). Under
this simplification and the small-game regime tested in Ch. 11, the bonus
is a null result — see `experiments/ch11_diversity/NOTES.md`.

## Adding a new row

When a new variant lands in this repo:

1. Mark ● in the columns where the variant substantively changes an
   existing component (Oracle, MSS, Evaluator, Loop). Mark ○ for
   inherited defaults.
2. Set **In repo** to ✔ or ✘. If ✘, keep it here anyway with a pointer
   to the section of the book that discusses why it's not implemented.
3. Set **Repo location** to the module(s) that carry the change, using
   paths relative to the repo root.
4. If the implementation departs from the paper's formulation in a way
   that affects claims, add a caveat entry above rather than hiding the
   delta in the row.

## Verification note

The ✘ rows (JPSRO, PSD-PSRO, XDO, TE-PSRO, anytime, Fusion-init) were
classified from the literature rather than from code. If anything in
that classification looks wrong when cross-checked against the 2024
Bighashdel et al. PSRO survey (see `docs/references.md`), the manuscript
should be corrected — Table 9.1 is printed in the book, so a
mis-classification here is a manuscript bug, not a repo bug.

# References

Rendered from `docs/bibliography.bib`. Grouped by section so a reader can
find where an author-year citation in the manuscript came from. Every
citation in the book text should appear below; if you spot one that
doesn't, please open an issue.

## Foundations of game theory

- **Brown, G. W.** (1951). *Iterative Solution of Games by Fictitious Play*.
  In T. C. Koopmans (ed.), *Activity Analysis of Production and Allocation*
  (pp. 374–376). Wiley.
- **Kuhn, H. W.** (1950). *A Simplified Two-Person Poker*. In H. W. Kuhn &
  A. W. Tucker (eds.), *Contributions to the Theory of Games, Volume 1*
  (Annals of Mathematics Studies 24, pp. 97–103). Princeton University
  Press.
- **Nash, J.** (1951). *Non-Cooperative Games*. *Annals of Mathematics*,
  54(2), 286–295. [DOI: 10.2307/1969529](https://doi.org/10.2307/1969529)
- **Robinson, J.** (1951). *An Iterative Method of Solving a Game*. *Annals
  of Mathematics*, 54(2), 296–301.
  [DOI: 10.2307/1969530](https://doi.org/10.2307/1969530)
- **Shapley, L. S.** (1964). *Some Topics in Two-Person Games*. In
  M. Dresher, L. S. Shapley, & A. W. Tucker (eds.), *Advances in Game
  Theory* (Annals of Mathematics Studies 52, pp. 1–28). Princeton
  University Press.
- **Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N.,
  Billings, D., & Rayner, C.** (2005). *Bayes' Bluff: Opponent Modelling
  in Poker*. In *Proceedings of the 21st Conference on Uncertainty in
  Artificial Intelligence (UAI)* (pp. 550–558). *(The paper that introduced
  Leduc hold'em as a reduced-form benchmark.)*
- **von Neumann, J.** (1928). *Zur Theorie der Gesellschaftsspiele*.
  *Mathematische Annalen*, 100, 295–320.
  [DOI: 10.1007/BF01448847](https://doi.org/10.1007/BF01448847)

## Double oracle and the PSRO family

- **Bighashdel, A., Wang, Y., McAleer, S., Savani, R., & Oliehoek, F. A.**
  (2024). *Policy Space Response Oracles: A Survey*. In *Proceedings of
  the 33rd International Joint Conference on Artificial Intelligence
  (IJCAI)*. [arXiv:2403.02227](https://arxiv.org/abs/2403.02227) /
  [DOI: 10.24963/ijcai.2024/880](https://doi.org/10.24963/ijcai.2024/880)
- **Heinrich, J., Lanctot, M., & Silver, D.** (2015). *Fictitious
  Self-Play in Extensive-Form Games*. In *Proceedings of the 32nd ICML*
  (pp. 805–813).
- **Lanctot, M., Zambaldi, V., Gruslys, A., Lazaridou, A., Tuyls, K.,
  Pérolat, J., Silver, D., & Graepel, T.** (2017). *A Unified
  Game-Theoretic Approach to Multiagent Reinforcement Learning*. In
  *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
  *(The original PSRO paper; source of NashConv as an exploitability proxy.)*
- **Marris, L., Muller, P., Lanctot, M., Tuyls, K., & Graepel, T.** (2021).
  *Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium
  Meta-Solvers*. In *Proceedings of the 38th ICML* (PMLR 139, pp.
  7480–7491). [arXiv:2106.09435](https://arxiv.org/abs/2106.09435).
  *(JPSRO / MGCE.)*
- **McAleer, S., Lanier, J., Baldi, P., & Fox, R.** (2021). *XDO: A Double
  Oracle Algorithm for Extensive-Form Games*. In *NeurIPS 34*.
  [arXiv:2103.06426](https://arxiv.org/abs/2103.06426)
- **McAleer, S., Lanier, J., Fox, R., & Baldi, P.** (2020). *Pipeline
  PSRO: A Scalable Approach for Finding Approximate Nash Equilibria in
  Large Games*. In *NeurIPS 33*.
  [arXiv:2006.08555](https://arxiv.org/abs/2006.08555)
- **McMahan, H. B., Gordon, G. J., & Blum, A.** (2003). *Planning in the
  Presence of Cost Functions Controlled by an Adversary*. In *Proceedings
  of the 20th ICML* (pp. 536–543). *(The double-oracle paper PSRO
  generalises.)*
- **Yao, J., Liu, W., Fu, H., Yang, Y., McAleer, S., Fu, Q., & Yang, W.**
  (2023). *Policy Space Diversity for Non-Transitive Games*. In *NeurIPS
  36*. [arXiv:2306.16884](https://arxiv.org/abs/2306.16884). *(PSD-PSRO.)*

## Meta-solvers and equilibrium concepts

- **Hart, S., & Mas-Colell, A.** (2000). *A Simple Adaptive Procedure
  Leading to Correlated Equilibrium*. *Econometrica*, 68(5), 1127–1150.
  *(Regret matching → CE/CCE.)*
- **Muller, P., Omidshafiei, S., Rowland, M., Tuyls, K., Pérolat, J.,
  Liu, S., Hennes, D., Marris, L., Lanctot, M., Hughes, E., Wang, Z.,
  Lever, G., Heess, N., Graepel, T., & Munos, R.** (2020). *A Generalized
  Training Approach for Multiagent Learning*. In *Proceedings of the
  8th ICLR*. [arXiv:1909.12823](https://arxiv.org/abs/1909.12823).
  *(α-Rank as a PSRO meta-solver.)*
- **Omidshafiei, S., Papadimitriou, C., Piliouras, G., Tuyls, K., Rowland,
  M., Lespiau, J.-B., Czarnecki, W. M., Lanctot, M., Pérolat, J., &
  Munos, R.** (2019). *α-Rank: Multi-Agent Evaluation by Evolution*.
  *Scientific Reports*, 9, 9937.
  [DOI: 10.1038/s41598-019-45619-9](https://doi.org/10.1038/s41598-019-45619-9)

## Diversity, gamescapes, geometry of games

- **Balduzzi, D., Garnelo, M., Bachrach, Y., Czarnecki, W. M., Pérolat,
  J., Jaderberg, M., & Graepel, T.** (2019). *Open-Ended Learning in
  Symmetric Zero-Sum Games*. In *Proceedings of the 36th ICML* (PMLR 97,
  pp. 434–443). [arXiv:1901.08106](https://arxiv.org/abs/1901.08106).
  *(Empirical gamescape; rectified Nash response.)*
- **Czarnecki, W. M., Gidel, G., Tracey, B., Tuyls, K., Omidshafiei, S.,
  Balduzzi, D., & Jaderberg, M.** (2020). *Real World Games Look Like
  Spinning Tops*. In *NeurIPS 33*.
  [arXiv:2004.09468](https://arxiv.org/abs/2004.09468). *(Transitive spine
  + cyclic radius geometry underlying PSRO's diversity story.)*

## Complexity of equilibrium computation

- **Chen, X., & Deng, X.** (2006). *Settling the Complexity of Two-Player
  Nash Equilibrium*. In *FOCS 47* (pp. 261–270).
  [DOI: 10.1109/FOCS.2006.69](https://doi.org/10.1109/FOCS.2006.69).
  *(PPAD-completeness for bimatrix.)*
- **Daskalakis, C., Goldberg, P. W., & Papadimitriou, C. H.** (2009).
  *The Complexity of Computing a Nash Equilibrium*. *SIAM Journal on
  Computing*, 39(1), 195–259.
  [DOI: 10.1137/070699652](https://doi.org/10.1137/070699652). *(Journal
  version of the STOC 2006 result establishing PPAD-completeness for
  many-player Nash.)*
- **Daskalakis, C., & Pan, Q.** (2014). *A Counter-Example to Karlin's
  Strong Conjecture for Fictitious Play*. In *FOCS 55* (pp. 11–20).
  [arXiv:1412.4840](https://arxiv.org/abs/1412.4840). *(FP's worst-case
  rate can be as slow as t^{-1/n} — the motivation for going beyond FP.)*

## Learning algorithms

- **Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P.**
  (2016). *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation*. In *ICLR 4*.
  [arXiv:1506.02438](https://arxiv.org/abs/1506.02438). *(GAE, used inside
  the book's single-file PPO oracle.)*
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.**
  (2017). *Proximal Policy Optimization Algorithms*.
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
- **Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C.** (2007).
  *Regret Minimization in Games with Incomplete Information*. In *NIPS 20*.
  *(CFR; the alternative solution family the book contrasts PSRO against.)*

## Systems and background

- **Albrecht, S. V., Christianos, F., & Schäfer, L.** (2024). *Multi-Agent
  Reinforcement Learning: Foundations and Modern Approaches*. MIT Press.
  [marl-book.com](https://www.marl-book.com). *(The MARL textbook the book
  refers readers to for RL-side background.)*
- **Amdahl, G. M.** (1967). *Validity of the Single Processor Approach to
  Achieving Large Scale Computing Capabilities*. In *Proceedings of the
  AFIPS Spring Joint Computer Conference* (pp. 483–485).
  [DOI: 10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560).
  *(Cited in Ch. 10's parallel-PSRO analysis.)*
- **Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E.,
  Elibol, M., Yang, Z., Paul, W., Jordan, M. I., & Stoica, I.** (2018).
  *Ray: A Distributed Framework for Emerging AI Applications*. In *OSDI 13*
  (pp. 561–577). *(The runtime used for parallel best-response training in
  Ch. 10.)*

## Unverified

None. Every entry above was cross-checked against a primary or canonical
source (publisher DOI, arXiv, or conference proceedings) during the M6+1
pass. If a citation in the manuscript does not resolve here, please open
an issue against this repository so the entry can be added.

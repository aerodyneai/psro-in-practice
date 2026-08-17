"""Shared ProcessPoolExecutor fan-out for embarrassingly-parallel PSRO studies.

Extracted after the ch12 shootout's parallel launcher shipped in brief-2 —
the same pattern is used by ch06 budget-split, ch06 BR-quality, ch12
warm-start, and any future study that dispatches many independent
``(condition, seed)`` runs. Each worker gets pinned to one BLAS/torch
thread via ``worker_init`` so N workers cleanly share N cores without
oversubscription.

Design notes:
  * Task tuples pickle across process boundaries — keep them plain (str,
    int, float). Passing an ``argparse.Namespace`` doesn't pickle
    reliably on 3.14; pass ``vars(args)`` and reconstruct inside the
    worker.
  * Results are written to disk after each future completes so a killed
    run leaves a valid partial CSV — the caller's next invocation
    filters ``tasks`` against what's already done and picks up where it
    stopped.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed


def worker_init() -> None:
    """Pin BLAS and torch to 1 thread per worker so N workers ≪ physical cores."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def fan_out(
    tasks: list[tuple],
    worker_fn: Callable[..., dict],
    args_dict: dict,
    n_workers: int,
    on_result: Callable[[dict, int, int], None] | None = None,
) -> list[dict]:
    """Parallelize task execution.

    Args:
        tasks: list of tuples; each tuple is spread as positional args to
            ``worker_fn`` followed by ``args_dict``.
        worker_fn: called as ``worker_fn(*task, args_dict) -> row_dict``.
            Must be top-level (picklable).
        args_dict: read-only dict passed to every worker call. Typically
            ``vars(argparse.Namespace)``.
        n_workers: ≥1. If 1, runs serially in the current process (useful
            for --smoke and single-machine debugging).
        on_result: optional callback ``(row, done_count, total) -> None``
            invoked after each completion. Persist to CSV here.

    Returns:
        List of result dicts, in completion order (not task order).
    """
    results: list[dict] = []
    if n_workers <= 1:
        for i, task in enumerate(tasks):
            row = worker_fn(*task, args_dict)
            results.append(row)
            if on_result:
                on_result(row, i + 1, len(tasks))
        return results
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=worker_init,
    ) as pool:
        futures = {pool.submit(worker_fn, *task, args_dict): task for task in tasks}
        done_count = 0
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            done_count += 1
            if on_result:
                on_result(row, done_count, len(tasks))
    return results

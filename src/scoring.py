"""Macro-averaged F_0.5 exactly as defined by the challenge, plus the decision
rule that turns pairwise match probabilities into per-entity match lists."""
from __future__ import annotations

import numpy as np
import polars as pl


def f05(prec: float, rec: float) -> float:
    if prec == 0 and rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05(pred: dict[str, set], truth: dict[str, set]) -> float:
    """pred / truth: source1_id -> set of matched ids. Every key of ``truth`` is scored."""
    tot = 0.0
    for s1, t in truth.items():
        p = pred.get(s1, set())
        if not t and not p:
            tot += 1.0
            continue
        if not t or not p:
            continue
        tp = len(p & t)
        tot += f05(tp / len(p), tp / len(t))
    return tot / max(len(truth), 1)


def decide(scored: pl.DataFrame, threshold: float, one_to_one: bool = True,
           min_gap: float = 0.0) -> dict[str, set]:
    """scored: s1_id, cand_id, p.  Returns s1_id -> set(cand_id).

    * keep pairs with p >= threshold
    * one_to_one: a Source 2/3 record may be claimed by at most one Source 1 entity
      (the one with the highest probability)
    """
    df = scored.filter(pl.col("p") >= threshold)
    if one_to_one and df.height:
        df = df.sort("p", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)
    out: dict[str, set] = {}
    for s1, c in zip(df["s1_id"].to_list(), df["cand_id"].to_list()):
        out.setdefault(s1, set()).add(c)
    return out


def decide_frame(scored: pl.DataFrame, threshold: float, one_to_one: bool = True) -> pl.DataFrame:
    """Frame version of ``decide`` (columns s1_id, cand_id kept)."""
    df = scored.filter(pl.col("p") >= threshold)
    if one_to_one and df.height:
        df = df.sort("p", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)
    return df.select("s1_id", "cand_id")


GROUP_KEY_COLS = ["b_addr_missing", "amb_name", "n_cands"]


def group_key(df: pl.DataFrame) -> pl.Series:
    """Small categorical key for per-group thresholds: address missing x ambiguous name x single candidate."""
    return ((pl.col("b_addr_missing") > 0).cast(pl.Int32) * 4
            + (pl.col("amb_name") > 1).cast(pl.Int32) * 2
            + (pl.col("n_cands") <= 1).cast(pl.Int32)).alias("g")


def decide_frame_groups(scored: pl.DataFrame, thresholds: dict, default: float, one_to_one: bool = True) -> pl.DataFrame:
    """Like decide_frame but with a threshold per group key (column 'g')."""
    thr = pl.col("g").cast(pl.Utf8).replace_strict({str(k): float(v) for k, v in thresholds.items()},
                                                    default=default, return_dtype=pl.Float64)
    df = scored.filter(pl.col("p") >= thr)
    if one_to_one and df.height:
        df = df.sort("p", descending=True).unique(subset=["cand_id"], keep="first", maintain_order=True)
    return df.select("s1_id", "cand_id")


def tune_group_thresholds(scored: pl.DataFrame, truth: dict, base: float, one_to_one: bool,
                          grid=None, passes: int = 2) -> tuple[dict, float]:
    """Greedy coordinate search of one threshold per group key, starting from the global optimum."""
    import numpy as np
    grid = np.arange(0.3, 0.96, 0.05) if grid is None else grid
    groups = sorted(scored["g"].unique().to_list())
    thr = {g: base for g in groups}
    best = macro_f05(decide_frame_to_dict(decide_frame_groups(scored, thr, base, one_to_one)), truth)
    for _ in range(passes):
        for g in groups:
            for t in grid:
                cand = dict(thr); cand[g] = float(t)
                f = macro_f05(decide_frame_to_dict(decide_frame_groups(scored, cand, base, one_to_one)), truth)
                if f > best + 1e-6:
                    best, thr = f, cand
    return thr, best


def decide_frame_to_dict(df: pl.DataFrame) -> dict:
    out = {}
    for a, b in zip(df["s1_id"].to_list(), df["cand_id"].to_list()):
        out.setdefault(int(a), set()).add(int(b))
    return out


def truth_from_pairs(gp: pl.DataFrame, all_a) -> dict:
    """gp: (a_idx, b_idx) true pairs; all_a: iterable of a_idx to score (singletons included)."""
    out = {int(x): set() for x in all_a}
    for x, y in zip(gp["a_idx"].to_list(), gp["b_idx"].to_list()):
        out.setdefault(int(x), set()).add(int(y))
    return out


def truth_from_gt(gt: pl.DataFrame, s1_ids) -> dict[str, set]:
    g = gt.filter(pl.col("source1_entity_id").is_in(s1_ids)).fill_null("")
    return {s: set(x for x in m.split(",") if x) for s, m in zip(g["source1_entity_id"], g["matched_entity_ids"])}


def sweep(scored: pl.DataFrame, truth: dict[str, set], thresholds=None, one_to_one=True) -> list[tuple[float, float]]:
    thresholds = thresholds if thresholds is not None else np.arange(0.20, 0.96, 0.05)
    res = []
    for t in thresholds:
        res.append((float(t), macro_f05(decide(scored, float(t), one_to_one), truth)))
    return res

"""Second-stage (stacking) model.

The first LightGBM scores every candidate pair independently.  The second stage
re-scores each pair using the first model's probabilities *in context*: how the
probability compares with the entity's best and second-best candidate, how many
candidates the entity has above 0.5, and how strongly competing entities claim the
same record.  This is what decides whether an ambiguous single candidate should be
matched or the entity left as a singleton.

Training data: the validation split of the first model (its probabilities are
out-of-sample there).  The second model is fitted with entity-grouped 4-fold CV to
pick the decision threshold, then refitted on all validation rows.
"""
from __future__ import annotations

import glob
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from block import gt_pairs_idx
from scoring import decide, macro_f05, sweep, truth_from_pairs

# base features carried through test scoring so that the second stage can use them
STACK_BASE_COLS = ["b_addr_missing", "amb_name", "n_cands", "combo", "combo_gap", "a_tsr", "n_tsr",
                   "house_rel", "n_cmp_eq", "n_both_eq", "twin_better", "b_deg", "b_combo_gap",
                   "sim_name", "sim_addr", "name_uniq_state", "n_hi_name", "n_hi_addr", "a_num_jacc",
                   "amb_addr", "n_first_eq", "state_rel"]
P_COLS = ["p", "p_max", "p_gap", "p_rank", "p_second", "p_sum", "n_p50", "n_p90", "p_logit",
          "b_p_max", "b_p_gap", "b_p_rank", "b_n_p50", "b_p_sum"]

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=8)


def stack_features(df: pl.DataFrame) -> pl.DataFrame:
    """df must have a_idx, b_idx, p (first-stage probability)."""
    top2 = (df.group_by("a_idx").agg(pl.col("p").sort(descending=True).head(2).alias("t"))
              .with_columns(p_second=pl.col("t").list.get(1, null_on_oob=True).fill_null(0.0)).drop("t"))
    df = df.join(top2, on="a_idx", how="left")
    p = pl.col("p").cast(pl.Float64)
    df = df.with_columns(
        p_max=p.max().over("a_idx"),
        p_rank=p.rank("min", descending=True).over("a_idx").cast(pl.Float32),
        p_sum=p.sum().over("a_idx"),
        n_p50=(p >= 0.5).sum().over("a_idx").cast(pl.Float32),
        n_p90=(p >= 0.9).sum().over("a_idx").cast(pl.Float32),
        p_logit=(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).log(),
        b_p_max=p.max().over("b_idx"),
        b_p_rank=p.rank("min", descending=True).over("b_idx").cast(pl.Float32),
        b_n_p50=(p >= 0.5).sum().over("b_idx").cast(pl.Float32),
        b_p_sum=p.sum().over("b_idx"),
    ).with_columns(p_gap=pl.col("p_max") - p, b_p_gap=pl.col("b_p_max") - p)
    return df


def feature_cols(base_cols):
    return P_COLS + list(base_cols)


def fit(model_dir: str, feat_dirs: list[str], a_path: str, b_paths: list[str], gt_path: str, out_dir: str,
        seed: int = 7, val_frac: float = 0.2):
    os.makedirs(out_dir, exist_ok=True)
    vs = pl.read_parquet(f"{model_dir}/val_scored.parquet").rename({"s1_id": "a_idx", "cand_id": "b_idx"})
    vs = vs.with_columns(pl.col("a_idx").cast(pl.UInt32), pl.col("b_idx").cast(pl.UInt32))
    feats = pl.concat([pl.read_parquet(p, columns=["a_idx", "b_idx", "y"] + STACK_BASE_COLS)
                       for d in feat_dirs for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    df = vs.select("a_idx", "b_idx", "p").join(feats, on=["a_idx", "b_idx"], how="inner")
    del feats
    df = stack_features(df)
    cols = feature_cols(STACK_BASE_COLS)
    print(f"stack rows {df.height:,} (pos {df['y'].sum():,}), {len(cols)} features", flush=True)
    # truth for the validation entities (same split as train.py)
    a = pl.read_parquet(a_path, columns=["entity_id", "country"])
    b = pl.concat([pl.read_parquet(p, columns=["entity_id"]) for p in b_paths])
    gt = pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
    rng = np.random.default_rng(seed)
    val_ids = np.flatnonzero(rng.random(a.height) < val_frac)
    truth = truth_from_pairs(gt_pairs_idx(gt, a, b).filter(pl.col("a_idx").is_in(pl.Series(val_ids, dtype=pl.UInt32).implode())), val_ids)
    base_scored = df.select(s1_id="a_idx", cand_id="b_idx", p=pl.col("p").cast(pl.Float64))
    base_best = max(sweep(base_scored, truth, one_to_one=True), key=lambda x: x[1])
    print(f"first stage alone: thr={base_best[0]:.2f} macro F0.5={base_best[1]:.4f}", flush=True)
    # entity-grouped 4-fold out-of-fold predictions
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = ((df["a_idx"].to_numpy().astype(np.int64) * 2654435761) % 2**32) % 4
    oof = np.zeros(len(y))
    best_iters = []
    for k in range(4):
        tr, te = fold != k, fold == k
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=cols)
        dte = lgb.Dataset(X[te], y[te], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dte],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        print(f"  fold {k}: best_iter {m.best_iteration}", flush=True)
    scored = df.select(s1_id="a_idx", cand_id="b_idx").with_columns(p=pl.Series(oof))
    best = (0.5, 0.0, True)
    for one in (True, False):
        for t, f in sweep(scored, truth, one_to_one=one):
            if f > best[1]:
                best = (t, f, one)
    for t, f in sweep(scored, truth, thresholds=np.arange(max(0.05, best[0] - 0.05), min(0.99, best[0] + 0.05), 0.01), one_to_one=best[2]):
        if f > best[1]:
            best = (t, f, best[2])
    print(f"STACK: thr={best[0]:.2f} one_to_one={best[2]} macro F0.5={best[1]:.4f}  (first stage {base_best[1]:.4f})", flush=True)
    pred = decide(scored, best[0], best[2])
    ctry = a["country"].to_list()
    for c in sorted(set(ctry)):
        sub = {k: v for k, v in truth.items() if ctry[k] == c}
        print(f"  {c}: macro F0.5 = {macro_f05(pred, sub):.4f}", flush=True)
    n_round = int(np.mean(best_iters) * 1.1)
    final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols), num_boost_round=n_round)
    final.save_model(f"{out_dir}/model2.txt")
    imp = sorted(zip(cols, final.feature_importance("gain")), key=lambda x: -x[1])
    json.dump({"threshold": best[0], "one_to_one": best[2], "val_macro_f05": best[1], "first_stage_f05": base_best[1],
               "base_cols": STACK_BASE_COLS, "rounds": n_round,
               "feature_importance": [(k, float(v)) for k, v in imp]}, open(f"{out_dir}/stack_config.json", "w"), indent=1)
    print("top stack features:", [(k, round(v / imp[0][1], 3)) for k, v in imp[:12]])


def apply(stack_dir: str, scored: pl.DataFrame) -> pl.DataFrame:
    """scored: a_idx, b_idx, p + STACK_BASE_COLS -> adds p2."""
    cfg = json.load(open(f"{stack_dir}/stack_config.json"))
    m = lgb.Booster(model_file=f"{stack_dir}/model2.txt")
    df = stack_features(scored)
    X = df.select(feature_cols(cfg["base_cols"])).to_numpy().astype(np.float32)
    return df.with_columns(p2=pl.Series(m.predict(X, num_threads=PARAMS["num_threads"]), dtype=pl.Float32))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    fit(args.model_dir, args.features, args.a, args.b, args.gt, args.out_dir)

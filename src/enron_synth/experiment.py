"""Experiment driver.

    python -m enron_synth.experiment select   --n 120
    python -m enron_synth.experiment run      --arm hybrid --model openai/gpt-4.1-mini
    python -m enron_synth.experiment run      --arm rule
    python -m enron_synth.experiment run      --arm llm_only
    python -m enron_synth.experiment evaluate --arms hybrid rule llm_only

Artifacts:  data/eval_set.jsonl, data/runs/<arm>.jsonl, reports/results.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluate import (Judge, build_people_fullnames, cohen_kappa, distinct_n, fidelity_frame, fidelity_summary,
                       privacy_row, render_real)
from .llm import LLM
from .parsing import parse_email
from .pipeline import DEFAULT_MODEL, SyntheticEmailGenerator

ROOT = Path(__file__).resolve().parents[2]
DATA, RUNS, REPORTS = ROOT / "data", ROOT / "data" / "runs", ROOT / "reports"


# ----------------------------------------------------------------------------------------- selection
def select(n: int = 120, seed: int = 11):
    """Stratified sample from the 30k-email sample: folder group x has-quoted-block, body 8-700 words."""
    df = pd.read_parquet(DATA / "sample.parquet")
    rows = []
    for f, m in zip(df["file"], df["message"]):
        e = parse_email(m, f)
        w = len(e.body.split())
        if not (8 <= w <= 700) or not e.to_addrs or not e.date or not e.from_addr:
            continue
        grp = "sent" if "sent" in e.folder_type else ("inbox" if "inbox" in e.folder_type else "other")
        rows.append({"file": f, "message": m, "grp": grp, "has_trailer": bool(e.trailer)})
    pool = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    per = max(n // 6, 1)
    picked = []
    for (_, _), g in pool.groupby(["grp", "has_trailer"]):
        picked.append(g.sample(min(per, len(g)), random_state=int(rng.integers(1e6))))
    ev = pd.concat(picked).sample(frac=1, random_state=seed).head(n)
    ev.to_json(DATA / "eval_set.jsonl", orient="records", lines=True)
    print(f"eval set: {len(ev)} emails from pool of {len(pool)}\n", ev.groupby(["grp", "has_trailer"]).size())


def load_eval() -> pd.DataFrame:
    return pd.read_json(DATA / "eval_set.jsonl", lines=True)


# ----------------------------------------------------------------------------------------- run
def run_arm(arm: str, model: str, n: int | None = None, workers: int = 8, name: str | None = None, temperature: float = 0.7):
    RUNS.mkdir(parents=True, exist_ok=True)
    ev = load_eval().head(n) if n else load_eval()
    gen = SyntheticEmailGenerator(model=model, temperature=temperature)
    mode = {"hybrid": "hybrid", "rule": "rule", "llm_only": "llm_only"}[arm]
    out_path = RUNS / f"{name or arm}.jsonl"
    t0 = time.time()

    def work(i):
        r = ev.iloc[i]
        try:
            s = gen.generate(r["message"], r["file"], mode=mode, seed=str(i))
            d = s.to_dict()
            d["idx"] = i
            d["file"] = r["file"]
            return d
        except Exception as ex:  # noqa: BLE001
            return {"idx": i, "error": repr(ex)}

    results = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(ev))]
        for f in tqdm(as_completed(futs), total=len(futs), desc=arm):
            results.append(f.result())
    results.sort(key=lambda d: d["idx"])
    with open(out_path, "w") as fh:
        for d in results:
            fh.write(json.dumps(d) + "\n")
    errs = sum("error" in d for d in results)
    print(f"{arm}: {len(results)-errs}/{len(results)} ok in {time.time()-t0:.0f}s | {LLM.report()}")


# ----------------------------------------------------------------------------------------- evaluate
def _load_run(name: str) -> dict[int, dict]:
    return {d["idx"]: d for d in map(json.loads, open(RUNS / f"{name}.jsonl")) if "error" not in d}


def evaluate(arms: list[str], n_judge: int | None = None, workers: int = 8, out: str = "results.json", judge_on: bool = True):
    ev = load_eval()
    people = build_people_fullnames(pd.read_parquet(DATA / "sample.parquet")["message"].tolist())
    judge = Judge() if judge_on else None
    gen_for_names = SyntheticEmailGenerator()      # deterministic scrub => recover synthetic people for every arm
    results = {}
    for arm in arms:
        run = _load_run(arm)
        idxs = sorted(run)[:n_judge] if n_judge else sorted(run)
        origs = [parse_email(ev.iloc[i]["message"], ev.iloc[i]["file"]) for i in idxs]
        syns = [run[i] for i in idxs]

        # 1. privacy  (synthetic people are recomputed deterministically so every arm is scored identically)
        for o, s in zip(origs, syns):
            if "synthetic_people" not in s["meta"]:
                from .personas import COMPANY_BY_ID, PersonaFactory
                from .pii import Scrubber
                sc = Scrubber(gen_for_names.factory(COMPANY_BY_ID[s["company"]])).scrub(o)
                s["meta"]["synthetic_people"] = sorted(sc.synthetic_names)
        pr = pd.DataFrame([privacy_row(o, s, people) for o, s in zip(origs, syns)])
        privacy = {
            "n": len(pr),
            "any_leak_rate": float(pr["any_leak"].mean()),
            "verifier_hard_leak_rate": float(pr["verifier_hard_leak"].mean()),
            "global_leak_rate": float(pr["global_leak"].mean()),
            "enron_person_fullname_rate": float(pr["enron_person_fullname"].mean()),
            "forced_redaction_rate": float(pr["forced_redaction"].mean()),
            "placeholder_left_rate": float(pr["placeholder_left"].mean()),
            "verbatim_8gram_overlap_mean": float(pr["verbatim_8gram_overlap"].mean()),
            "verbatim_8gram_overlap_p95": float(pr["verbatim_8gram_overlap"].quantile(.95)),
            "emails_with_>10%_verbatim": float((pr["verbatim_8gram_overlap"] > 0.10).mean()),
        }
        # 2. fidelity
        fdf = fidelity_frame(origs, syns)
        fid = fidelity_summary(fdf)
        fid["distinct1_syn"] = distinct_n([s["body"] for s in syns], 1)
        fid["distinct2_syn"] = distinct_n([s["body"] for s in syns], 2)
        fid["distinct1_orig"] = distinct_n([o.body for o in origs], 1)
        fid["distinct2_orig"] = distinct_n([o.body for o in origs], 2)

        # 3 + 4. LLM judging (parallel)
        def judge_one(k):
            o, s = origs[k], syns[k]
            real_txt, syn_txt = render_real(o), s["raw"]
            rr = judge.realism(syn_txt)
            ro = judge.realism(real_txt)
            pw = judge.pairwise(real_txt, syn_txt, real_first=(k % 2 == 0))
            lo, ls = judge.labels(real_txt), judge.labels(syn_txt)
            return {"realism_syn": rr.get("realism"), "realism_real": ro.get("realism"),
                    "artifacts": rr.get("artifacts", []), "pw": pw, "lab_o": lo, "lab_s": ls}

        if judge_on:
            with ThreadPoolExecutor(workers) as ex:
                js = list(tqdm(ex.map(judge_one, range(len(idxs))), total=len(idxs), desc=f"judge {arm}"))
        else:
            js = [{"realism_syn": None, "realism_real": None, "artifacts": [], "pw": {"correct": None, "picked": None, "reason": ""},
                   "lab_o": {}, "lab_s": {}} for _ in idxs]
        rs = [j["realism_syn"] for j in js if isinstance(j["realism_syn"], (int, float))]
        ro = [j["realism_real"] for j in js if isinstance(j["realism_real"], (int, float))]
        pw = [j["pw"]["correct"] for j in js if j["pw"]["picked"] in ("A", "B")]
        realism = {
            "realism_syn_mean": float(np.mean(rs)) if rs else None, "realism_real_mean": float(np.mean(ro)) if ro else None,
            "judge_detects_synthetic_acc": float(np.mean(pw)) if pw else None,
            "n_pairs": len(pw),
            "top_artifacts": _top([a for j in js for a in j["artifacts"]]),
        }
        util = {}
        for fld in ["category", "intent", "sentiment", "privileged_likely", "action_required", "sensitive"]:
            a = [j["lab_o"].get(fld) for j in js if fld in j["lab_o"] and fld in j["lab_s"]]
            b = [j["lab_s"].get(fld) for j in js if fld in j["lab_o"] and fld in j["lab_s"]]
            util[fld] = {"agreement": float(np.mean([x == y for x, y in zip(a, b)])) if a else None,
                         "kappa": float(cohen_kappa(a, b)) if a else None}
        # meta
        meta = {"mean_attempts": float(np.mean([s["meta"].get("attempts", 0) for s in syns])),
                "trailer_truncated": float(np.mean([s["meta"].get("trailer_truncated", False) for s in syns])),
                "companies": pd.Series([s["company"] for s in syns]).value_counts().to_dict()}
        results[arm] = {"privacy": privacy, "fidelity": fid, "realism": realism, "utility": util, "meta": meta}
        (REPORTS).mkdir(exist_ok=True)
        pd.DataFrame([{**pr.iloc[k].to_dict(), **{"realism": js[k]["realism_syn"], "idx": idxs[k], "pw_correct": js[k]["pw"]["correct"],
                                                  "lab_o": js[k]["lab_o"], "lab_s": js[k]["lab_s"], "artifacts": js[k]["artifacts"]}} for k in range(len(idxs))]).to_json(
            REPORTS / f"per_email_{arm}.jsonl", orient="records", lines=True)
        print(arm, json.dumps({"privacy": privacy, "realism": realism}, indent=1)[:1500])
    (REPORTS / out).write_text(json.dumps(results, indent=2, default=str))
    print(LLM.report())


def _top(items: list[str], k: int = 5) -> list[str]:
    from collections import Counter
    return [f"{t} ({c})" for t, c in Counter(i.lower()[:90] for i in items).most_common(k)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("select"); a.add_argument("--n", type=int, default=120)
    r = sub.add_parser("run"); r.add_argument("--arm", required=True); r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--n", type=int); r.add_argument("--name"); r.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("evaluate"); e.add_argument("--arms", nargs="+", required=True); e.add_argument("--n", type=int); e.add_argument("--out", default="results.json"); e.add_argument("--workers", type=int, default=6); e.add_argument("--no-judge", action="store_true", help="skip LLM-judge metrics (no API needed)")
    args = ap.parse_args()
    if args.cmd == "select":
        select(args.n)
    elif args.cmd == "run":
        run_arm(args.arm, args.model, args.n, args.workers, args.name)
    else:
        evaluate(args.arms, args.n, workers=args.workers, out=args.out, judge_on=not args.no_judge)

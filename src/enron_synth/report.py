"""Tables + figures from the experiment artifacts.

    python -m enron_synth.report

Reads   reports/results_offline.json   (privacy + fidelity, no API needed, every arm)
        reports/results_judged_v1.json (LLM-judge realism + label preservation, arms judged with the v1 pipeline)
Writes  reports/results_offline.md, reports/results_judged.md, reports/figures/results_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NICE = {"hybrid_v1": "Hybrid v1 (GPT-4.1-mini)", "hybrid": "Hybrid v2 (GPT-4.1-mini)", "rule_v2": "Rule-based (v2 scrub)",
        "rule": "Rule-based (final scrub)", "llm_only": "LLM only (naive)",
        "hybrid_llama70b": "Hybrid v1 · Llama-3.3-70B", "hybrid_qwen235b": "Hybrid v1 · Qwen3-235B",
        "hybrid_gemini_flash": "Hybrid v1 · Gemini-2.5-Flash", "hybrid_haiku45": "Hybrid v1 · Claude Haiku 4.5"}
JUDGED_NICE = {"hybrid": "Hybrid v1 (GPT-4.1-mini)", "rule_v2": "Rule-based, scrubber as of v2", "rule": "Rule-based, final scrubber", "llm_only": "LLM only (naive)"}
COST = {"hybrid_v1": 0.0012, "hybrid": 0.0012, "llm_only": 0.0007, "hybrid_llama70b": 0.00045, "hybrid_qwen235b": 0.00055,
        "hybrid_gemini_flash": 0.0017, "hybrid_haiku45": 0.0043}      # USD / e-mail, from the OpenRouter usage logs of the runs


def offline_rows(res: dict) -> list[dict]:
    out = []
    for arm, v in res.items():
        p, f, m = v["privacy"], v["fidelity"], v["meta"]
        out.append({
            "key": arm, "arm": NICE.get(arm, arm), "n": p["n"], "any_leak": p["any_leak_rate"], "global": p["global_leak_rate"],
            "person": p["enron_person_fullname_rate"], "forced": p["forced_redaction_rate"], "copy8": p["verbatim_8gram_overlap_mean"],
            "copy10": p["emails_with_>10%_verbatim"], "rho": f["body_words_spearman"], "ks": f["body_words_ks_stat"],
            "struct": float(np.mean([f["has_greeting_agree"], f["has_signoff_agree"], f["has_trailer_agree"], f["is_re_agree"]])),
            "para": f["n_paragraphs_spearman"], "attempts": m["mean_attempts"], "cost": COST.get(arm),
        })
    return out


def offline_md(rs) -> str:
    h = ("| Arm | n | Any leak ↓ | Global Enron-marker ↓ | Enron full-name ↓ | Forced redaction | 8-gram copy ↓ | Length ρ ↑ | Length KS ↓ | Structure agree ↑ | LLM calls/e-mail | $/e-mail |\n"
         "|" + "---|" * 12 + "\n")
    return h + "".join(
        f"| {r['arm']} | {r['n']} | {r['any_leak']:.1%} | {r['global']:.1%} | {r['person']:.1%} | {r['forced']:.1%} | {r['copy8']:.1%} | "
        f"{r['rho']:.2f} | {r['ks']:.2f} | {r['struct']:.1%} | {r['attempts']:.2f} | {'$%.4f' % r['cost'] if r['cost'] else '-'} |\n" for r in rs)


def judged_md(res: dict) -> str:
    h = ("| Arm | n | Judge score (1-5) ↑ | Real e-mail score | Judge spots the synthetic one ↓ (50% = chance) | Category κ ↑ | Intent agree ↑ | Sentiment agree ↑ | "
         "Privilege agree ↑ | Action-required agree ↑ | Sensitive agree ↑ |\n|" + "---|" * 11 + "\n")
    b = ""
    for arm, v in res.items():
        r, u = v["realism"], v["utility"]
        b += (f"| {JUDGED_NICE.get(arm, arm)} | {v['privacy']['n']} | {r['realism_syn_mean']:.2f} | {r['realism_real_mean']:.2f} | {r['judge_detects_synthetic_acc']:.1%} | "
              f"{u['category']['kappa']:.2f} | {u['intent']['agreement']:.1%} | {u['sentiment']['agreement']:.1%} | {u['privileged_likely']['agreement']:.1%} | "
              f"{u['action_required']['agreement']:.1%} | {u['sensitive']['agreement']:.1%} |\n")
    return h + b


def figure(rs, judged, path: Path):
    main = [r for r in rs if r["key"] in ("hybrid_v1", "hybrid", "rule", "llm_only")]
    models = [r for r in rs if r["key"] in ("hybrid_v1", "hybrid_llama70b", "hybrid_qwen235b", "hybrid_gemini_flash", "hybrid_haiku45")]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    col = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    def bars(a, rows, key, title, fmt="{:.0%}", ylim=(0, 1)):
        vals = [r[key] for r in rows]
        a.bar(range(len(rows)), vals, color=col[: len(rows)])
        a.set_xticks(range(len(rows)), [r["arm"].replace(" (GPT-4.1-mini)", "").replace("Hybrid v1 · ", "") for r in rows], rotation=25, ha="right", fontsize=8)
        a.set_title(title, fontsize=10)
        a.set_ylim(*ylim)
        for i, v in enumerate(vals):
            a.text(i, v + 0.015 * (ylim[1] - ylim[0]), fmt.format(v), ha="center", fontsize=8)
    bars(ax[0, 0], main, "any_leak", "Privacy: e-mails with any leak (lower is better)")
    bars(ax[0, 1], main, "copy8", "Copying: mean verbatim 8-gram overlap with source (lower)")
    bars(ax[0, 2], main, "rho", "Fidelity: length Spearman ρ vs source (higher)", "{:.2f}", ylim=(0, 1.12))
    bars(ax[1, 0], main, "struct", "Fidelity: greeting/sign-off/quote/Re: agreement (higher)", ylim=(0, 1.12))
    bars(ax[1, 1], models, "any_leak", "Model comparison: any leak (same v1 pipeline)", ylim=(0, 0.2))
    bars(ax[1, 2], models, "copy8", "Model comparison: verbatim 8-gram overlap")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()

    if judged:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        arms = [JUDGED_NICE.get(a, a) for a in judged]
        for a, (t, vals, lim, f) in zip(ax, [
            ("LLM-judge realism score (1-5, real mail ≈ 4.0)", [v["realism"]["realism_syn_mean"] for v in judged.values()], (1, 5), "{:.2f}"),
            ("Judge picks the real e-mail (50% = indistinguishable)", [v["realism"]["judge_detects_synthetic_acc"] for v in judged.values()], (0, 1), "{:.0%}"),
            ("Label preservation: category Cohen's κ", [v["utility"]["category"]["kappa"] for v in judged.values()], (0, 1), "{:.2f}")]):
            a.bar(range(len(arms)), vals, color=col[:3])
            a.set_xticks(range(len(arms)), arms, rotation=15, ha="right", fontsize=8)
            a.set_ylim(*lim)
            a.set_title(t, fontsize=10)
            for i, v in enumerate(vals):
                a.text(i, v + 0.02 * (lim[1] - lim[0]), f.format(v), ha="center", fontsize=9)
        ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
        plt.tight_layout()
        plt.savefig(path.with_name("results_judged.png"), dpi=130)
        plt.close()


EX_NICE = {"hybrid_v1": "Hybrid v1", "hybrid": "Hybrid v2", "rule_v2": "Rule-based, scrubber as of v2", "rule": "Rule-based, final scrubber", "llm_only": "LLM only (naive)",
           "hybrid_llama70b": "Llama-3.3-70B", "hybrid_qwen235b": "Qwen3-235B", "hybrid_gemini_flash": "Gemini-2.5-Flash", "hybrid_haiku45": "Claude Haiku 4.5"}


def extra_md(ex: dict) -> str:
    h = ("| Arm | Embedding cos (pair / random) | Own-source top-1 / top-5 | Form-classifier AUC raw → **styled** ↓ | Source names surviving (e-mails) ↓ | Sender in sign-off | Diversity distinct-2 |\n"
         "|" + "---|" * 7 + "\n")
    o = ex["originals"]
    b = f"| *Original Enron (reference)* | - | - | 0.50 by definition | - | {o['sign_ok']:.0%} | {o['distinct2']:.3f} |\n"
    for k, name in EX_NICE.items():
        if k not in ex:
            continue
        v, st = ex[k], ex.get(k + "_styled")
        auc = f"{v['form_discriminator_auc']:.2f} → **{st['form_discriminator_auc']:.2f}**" if st else f"{v['form_discriminator_auc']:.2f}"
        b += (f"| {name} | {v['embed_cos_pair_mean']:.2f} / {v['embed_cos_random_pair_mean']:.2f} | {v['embed_top1_retrieval']:.0%} / {v['embed_top5_retrieval']:.0%} | {auc} | "
              f"{v['source_name_survivor_email_rate']:.1%} ({v['source_name_survivors_total']} names) | {v['sign_ok']:.0%} | {v['distinct2']:.3f} |\n")
    return h + b


if __name__ == "__main__":
    rep = ROOT / "reports"
    off = json.load(open(rep / "results_offline.json"))
    rs = offline_rows(off)
    (rep / "results_offline.md").write_text(offline_md(rs))
    judged = json.load(open(rep / "results_judged_v1.json")) if (rep / "results_judged_v1.json").exists() else None
    if judged:
        (rep / "results_judged.md").write_text(judged_md(judged))
    if (rep / "results_extra.json").exists():
        (rep / "results_extra.md").write_text(extra_md(json.load(open(rep / "results_extra.json"))))
    (rep / "figures").mkdir(exist_ok=True)
    figure(rs, judged, rep / "figures" / "results_comparison.png")
    print(offline_md(rs)); print(judged_md(judged) if judged else "")

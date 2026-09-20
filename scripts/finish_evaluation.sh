#!/usr/bin/env sh
# Completes the LLM-dependent parts once the OpenRouter account has credit again (~US$2 in total):
#  1. regenerate the hybrid arm with the final scrubber (40/120 e-mails miss the cache, the rest are served from data/cache)
#  2. LLM-judge the final hybrid + baselines  (realism, pairwise detectability, label preservation)
#  3. LLM-judge the four alternative generator models on the first 40 e-mails
set -e
export PYTHONPATH=src SYNTH_MAX_CONCURRENCY=2
python -m enron_synth.experiment run --arm hybrid --workers 2
python -m enron_synth.experiment evaluate --arms hybrid rule llm_only --workers 2 --out results_judged_final.json
python -m enron_synth.experiment evaluate --arms hybrid_v1 hybrid_llama70b hybrid_qwen235b hybrid_gemini_flash hybrid_haiku45 --n 40 --workers 2 --out results_models_judged.json
python scripts/postprocess_run.py hybrid
python -m enron_synth.evaluate_extra --arms hybrid hybrid_styled rule llm_only
python -m enron_synth.report

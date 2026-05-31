import json, glob
import os

models = {
    "base": "results/lm_eval/lm_eval_base.json",
    "heretic": "results/lm_eval/lm_eval_heretic.json",
    "hauhau": "results/lm_eval/lm_eval_hauhau.json",
    "huihui": "results/lm_eval/lm_eval_huihui.json",
    "aeon": "results/lm_eval/lm_eval_aeon.json",
    "abliterix": "results/lm_eval/lm_eval_abliterix.json"
}

tasks = {
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc,none",
    "winogrande": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "piqa": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "lambada_openai": "perplexity,none"
}

print("Benchmarks:")
for model, path in models.items():
    if not os.path.exists(path):
        print(f"{model}: File not found {path}")
        continue
    with open(path) as f:
        data = json.load(f)
    res = data.get("results", {})
    scores = []
    for task, metric in tasks.items():
        if task in res:
            val = res[task].get(metric, 0)
            if "perplexity" not in metric:
                scores.append(f"{task}: {val*100:.1f}%")
            else:
                scores.append(f"{task}: {val:.2f}")
        else:
            scores.append(f"{task}: N/A")
    print(f"{model}: {', '.join(scores)}")

print("\nKL Divergence:")
kl_models = {
    "heretic": "results/kl/kl_heretic.json",
    "hauhau": "results/kl/kl_hauhau.json",
    "huihui": "results/kl/kl_huihui.json",
    "aeon": "results/kl/kl_aeon.json",
    "abliterix": "results/kl/kl_abliterix.json"
}
for model, path in kl_models.items():
    if not os.path.exists(path): continue
    with open(path) as f:
        data = json.load(f)
    print(f"{model}: {data.get('kl_divergence_batchmean')}")


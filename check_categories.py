import json

models = ["base", "huihui", "abliterix", "aeon", "heretic", "hauhau"]
categories = [
    "chemical_biological",
    "copyright",
    "cybercrime_intrusion",
    "harassment_bullying",
    "harmful",
    "illegal",
    "misinformation_disinformation"
]

for model in models:
    with open(f"results/qwen36-27b/harmbench/harmbench_{model}_scores.json") as f:
        data = json.load(f)
    print(f"Model: {model}")
    for cat in categories:
        asr = data["by_category"].get(cat, {}).get("asr", 0)
        print(f"  {cat}: {asr * 100:.1f}%")

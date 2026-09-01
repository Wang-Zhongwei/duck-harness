#!/usr/bin/env python3
"""Dump each run's system prompt and diff them, so a rate difference between
runs can be attributed to a prompt change (or ruled out)."""
import difflib, glob, json, os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "prompts")


def system_prompt(run_dir):
    ev = sorted(glob.glob(os.path.join(run_dir, "passes", "*", "artifacts", "*_events.jsonl")))[0]
    for line in open(ev):
        if '"type":"analysis"' not in line and '"type": "analysis"' not in line:
            continue
        t = json.loads(line)["transcript"]
        return t[t.find("[SYSTEM PROMPT]") + len("[SYSTEM PROMPT]"):t.find("[USER PROMPT]")].strip()
    return ""


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [os.path.basename(r.rstrip("/")) for r in sys.argv[1:]]
    texts = {}
    for run_dir, name in zip(sys.argv[1:], runs):
        texts[name] = system_prompt(run_dir)
        open(os.path.join(OUT, name + ".txt"), "w").write(texts[name])
        print(f"{name:58s} {len(texts[name]):6d} chars  (~{len(texts[name])//4} tok)")

    print("\n--- does each prompt still carry the world-model scaffolding? ---")
    keys = ["restate your working world model", "Helpful optional prefixes",
            "`World model:`", "`Action model:`", "Per-turn protocol"]
    for name in runs:
        hits = " ".join(f"{k.strip('`')}={texts[name].count(k)}" for k in keys)
        print(f"  {name[:50]:50s} {hits}")

    print("\n--- pairwise diffs vs the first run ---")
    base = runs[0]
    for name in runs[1:]:
        d = list(difflib.unified_diff(texts[base].split("\n"), texts[name].split("\n"),
                                      base, name, n=0, lineterm=""))
        path = os.path.join(OUT, f"diff_{base}__vs__{name}.txt")
        open(path, "w").write("\n".join(d))
        adds = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        print(f"  {name[:50]:50s} +{adds} -{dels} lines -> {os.path.relpath(path, BASE)}")


if __name__ == "__main__":
    main()

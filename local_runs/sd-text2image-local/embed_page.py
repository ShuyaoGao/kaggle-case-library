"""把本机运行汇总 results.json 和分词结果 tokens.json 写进网页 cases/sd-text2image-local.html。
用法：python -I -X utf8 embed_page.py <Kaggle_Cases 根目录>
"""
import json, os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
page = os.path.join(sys.argv[1], "cases", "sd-text2image-local.html")
runs = json.load(open(os.path.join(here, "results.json"), encoding="utf8"))
for r in runs.values():
    r.pop("timesteps", None)
    if len(r.get("prompt", "")) > 80:
        r["prompt"] = r["prompt"][:80] + "…"
toks = json.load(open(os.path.join(here, "tokens.json"), encoding="utf8"))
s = open(page, encoding="utf8").read()
s = re.sub(r"const RUNS = .*?;\n", lambda m: "const RUNS = " + json.dumps(runs, ensure_ascii=False, separators=(",", ":")) + ";\n", s, count=1)
s = re.sub(r"const TOKS = .*?;\n", lambda m: "const TOKS = " + json.dumps(toks, ensure_ascii=False, separators=(",", ":")) + ";\n", s, count=1)
open(page, "w", encoding="utf8").write(s)
print("ok", len(runs), "runs")

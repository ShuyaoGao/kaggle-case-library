"""把本机训练结果写进网页 cases/ddpm-mnist-scratch.html：
  - 训练信息（run_main.json、run_web.json）→ 页面脚本里的 `const RUN = ...;`
  - 浏览器版小网络的 float16 权重（weights_web.f16）→ <script id="ddpm-weights"> 里的 base64
  - 示意图 PNG → cases/img/ddpm-mnist-scratch/
用法：python -I -X utf8 embed_page.py <runs 目录（含 main/ 和 web/）> <Kaggle_Cases 根目录>
"""
import base64, json, os, re, shutil, sys

runs, root = sys.argv[1], sys.argv[2]
page = os.path.join(root, "cases", "ddpm-mnist-scratch.html")
img = os.path.join(root, "cases", "img", "ddpm-mnist-scratch")
os.makedirs(img, exist_ok=True)

main = json.load(open(os.path.join(runs, "main", "run_main.json"), encoding="utf8"))
web = json.load(open(os.path.join(runs, "web", "run_web.json"), encoding="utf8"))
shapes = json.load(open(os.path.join(runs, "web", "weights_web.json"), encoding="utf8"))
web.pop("loss_curve", None)
info = {"main": main, "web": web, "webShapes": shapes}
b64 = base64.b64encode(open(os.path.join(runs, "web", "weights_web.f16"), "rb").read()).decode()

s = open(page, encoding="utf8").read()
s = re.sub(r'(<script id="ddpm-weights" type="text/plain">)[^<]*(</script>)', lambda m: m.group(1) + b64 + m.group(2), s, count=1)
s = re.sub(r"const RUN = .*?;\n", lambda m: "const RUN = " + json.dumps(info, ensure_ascii=False, separators=(",", ":")) + ";\n", s, count=1)
open(page, "w", encoding="utf8").write(s)

for n in ["forward_main.png", "samples_main.png", "reverse_main.png"]:
    shutil.copy(os.path.join(runs, "main", n), os.path.join(img, n))
for n in ["run_main.json", "run_web.json"]:
    shutil.copy(os.path.join(runs, n.split("_")[1].split(".")[0], n), os.path.join(os.path.dirname(os.path.abspath(__file__)), n))
print("page bytes:", len(s.encode("utf8")), " weights b64:", len(b64))

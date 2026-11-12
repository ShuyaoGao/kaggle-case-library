"""把 sd_onnx_dml.py 的输出整理成网页用的图片（cases/img/sd-text2image-local/）和结果汇总 results.json。
用法：python -I -X utf8 make_figures.py <输出目录 out> <网页图片目录>
"""
import glob, json, os, sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)


def strip(files, size, path, cols=None):
    ims = [Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS) for f in files]
    cols = cols or len(ims)
    rows = (len(ims) + cols - 1) // cols
    W = Image.new("RGB", (cols * size + (cols - 1) * 4, rows * size + (rows - 1) * 4), "white")
    for i, im in enumerate(ims):
        W.paste(im, ((i % cols) * (size + 4), (i // cols) * (size + 4)))
    W.save(path, optimize=True)


J = lambda n: os.path.join(src, n)
# 单张大图
for n in ["sd15_cave_50.png", "sd15_cave_500.png", "turbo_cave_s1.png", "turbo_cave_s4.png", "turbo_synth_s1.png", "turbo_glass_s1.png"]:
    Image.open(J(n)).convert("RGB").resize((384, 384), Image.LANCZOS).save(os.path.join(dst, n), optimize=True)
Image.open(J("sd15_cave_50_latent.png")).save(os.path.join(dst, "sd15_cave_50_latent.png"), optimize=True)
# SD1.5 50 步：上排带噪潜变量，下排“猜测的成品”
steps = [1, 10, 20, 30, 40, 50]
strip([J(f"snap_sd15_cave_50/step{s:03d}_xt.png") for s in steps] + [J(f"snap_sd15_cave_50/step{s:03d}_x0.png") for s in steps], 160, os.path.join(dst, "sd15_steps.png"), cols=6)
# 引导强度对比
strip([J(f"sd15_synth_g{g}.png") for g in ["1", "3", "7.5", "15"]], 240, os.path.join(dst, "sd15_cfg.png"))
# SD-Turbo 4 步：每一步猜测的成品
strip([J(f"snap_turbo_cave_s4/step{s:03d}_x0.png") for s in [1, 2, 3, 4]], 240, os.path.join(dst, "turbo_steps.png"))
# 不同种子
strip([J(f"turbo_synth_seed{s}.png") for s in [1, 2, 3, 4]], 240, os.path.join(dst, "turbo_seeds.png"), cols=2)

res = {}
for f in sorted(glob.glob(J("*.json"))):
    d = json.load(open(f, encoding="utf8"))
    d.pop("sigmas", None)
    res[os.path.basename(f)[:-5]] = d
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(res), "runs;", sorted(os.listdir(dst)))

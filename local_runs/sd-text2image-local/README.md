# sd-text2image-local：本机运行记录

对应案例页 `cases/sd-text2image-local.html`，原 notebook：burhanuddinlatsaheb/text-to-image-generation-stable-diffusion。

## 路线

本机（RTX 5090 Laptop 24GB，Windows 11，Python 3.12）开着 Windows「智能应用控制」，PyTorch 导入不了，**没有关闭它**。走的路线：

- **ONNX Runtime + DirectML**（`onnxruntime-directml` 1.24.4，微软签名）在显卡上跑文本编码器、U-Net、VAE 解码器；
- 调度器循环（Euler、K-LMS）、无分类器引导、潜变量初始化都用 numpy 重写（`sd_onnx_dml.py`）；
- 分词用 `transformers` 的 CLIPTokenizer（不需要 PyTorch）。

试过但没走通：`onnxruntime/sd-turbo`（微软官方 ONNX）是为 CUDA 执行器优化的，含 `com.microsoft.SkipGroupNorm`、`NhwcConv` 等算子，DirectML 和 CPU 都报“找不到内核”，其 README 也写明只能用 CUDA。改用下面两份普通导出版。

## 模型（均为 Hugging Face 公开仓库，无需登录）

| 名称 | 仓库 | 说明 |
|---|---|---|
| SD v1.5 | `onnx-community/stable-diffusion-v1-5-ONNX`（CreativeML OpenRAIL-M） | 作者用 v1.4，v1.5 是其后续训练版；文本编码器同为 CLIP ViT-L/14 |
| SD-Turbo | `Heliosoph/sd-turbo-onnx`（原模型 stabilityai/sd-turbo，非商业研究许可） | fp32，1–4 步出图，不用 CFG |

只下载 `text_encoder`、`unet`、`vae_decoder`、`tokenizer`、配置文件（约 4 GB + 5 GB），放在临时目录，没有放进案例库。

## 环境

```
python -m venv venv
venv\Scripts\python -m pip install onnxruntime-directml numpy pillow huggingface_hub transformers==4.46.3
```

## 命令

```
set S=sd_onnx_dml.py
set P1=a curious explorer discovers a massive sprawling underground city ...（作者第 25 格全文）
python -I -X utf8 %S% --model sd15  --model_dir models\sd15 --prompt "%P1%" --steps 50  --seed 48 --out out\sd15_cave_50.png --snap_dir out\snap_sd15_cave_50 --snap_every 5
python -I -X utf8 %S% --model sd15  --model_dir models\sd15 --prompt "%P1%" --steps 500 --seed 48 --out out\sd15_cave_500.png
python -I -X utf8 %S% --model sd15  --model_dir models\sd15 --prompt "<作者第 52 格>" --steps 30 --guidance 1|3|7.5|15 --seed 48 --out out\sd15_synth_g<g>.png
python -I -X utf8 %S% --model turbo --model_dir models\sd-turbo-fp32 --prompt "%P1%" --steps 1 --seed 48 --out out\turbo_cave_s1.png
python -I -X utf8 %S% --model turbo --model_dir models\sd-turbo-fp32 --prompt "%P1%" --steps 4 --seed 48 --out out\turbo_cave_s4.png --snap_dir out\snap_turbo_cave_s4
python -I -X utf8 %S% --model turbo --model_dir models\sd-turbo-fp32 --prompt "<作者第 52 格>" --steps 1 --seed 48 --out out\turbo_synth_s1.png
python -I -X utf8 %S% --model turbo --model_dir models\sd-turbo-fp32 --prompt "<作者第 52 格>" --steps 2 --seed 1..4 --out out\turbo_synth_seed<k>.png
python -I -X utf8 %S% --model turbo --model_dir models\sd-turbo-fp32 --prompt "<作者第 54 格>" --steps 1 --seed 48 --out out\turbo_glass_s1.png
python -I -X utf8 tokens.py models\sd15
python -I -X utf8 make_figures.py out D:\教材\Kaggle_Cases\cases\img\sd-text2image-local
python -I -X utf8 embed_page.py D:\教材\Kaggle_Cases
```

每次运行都会在 png 旁边写一个同名 json（参数、词块数、是否截断、耗时）；全部汇总在 `results.json`。

## 结果与耗时（2026-11-12，512×512）

| 模型 | 提示词 | 步数 | 引导 | 种子 | 去噪循环 | 总耗时（含读模型） |
|---|---|---|---|---|---|---|
| SD v1.5 | 洞穴城市 | 500（作者设置） | 7.5 | 48 | 84.4 秒 | 92.4 秒 |
| SD v1.5 | 洞穴城市 | 50 | 7.5 | 48 | 12.3 秒 | 20.3 秒 |
| SD v1.5 | 合成波城市 | 30 | 1 / 3 / 7.5 / 15 | 48 | 3.3–6.5 秒 | 9–13 秒 |
| SD-Turbo | 洞穴城市 | 1 | 0 | 48 | 0.7 秒 | 8.5 秒 |
| SD-Turbo | 洞穴城市 | 4 | 0 | 48 | 2.0 秒（含每步解码中间图） | 12.6 秒 |
| SD-Turbo | 玻璃末日城市（139 词块，截断到 77） | 1 | 0 | 48 | 0.8 秒 | 12.3 秒 |

发现：50 步与 500 步的图几乎一样；作者第 54 格提示词有 139 个 CLIP 词块，后 62 个被截掉。
numpy 随机数与 PyTorch 不同，种子 48 不能复现作者的图。生成图片均为 AI 生成，仅作课堂演示。

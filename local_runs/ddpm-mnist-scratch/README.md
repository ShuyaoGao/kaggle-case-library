# ddpm-mnist-scratch：本机复现记录

对应案例页 `cases/ddpm-mnist-scratch.html`，原 notebook：vikramsandu/ddpm-from-scratch-in-pytorch。

## 为什么不用作者的 PyTorch 代码

本机（RTX 5090 Laptop，Windows 11，Python 3.12）开着 Windows「智能应用控制」，PyTorch 的未签名 DLL 会被拦截，导入不了。没有关闭这个安全功能，改为用 **纯 numpy** 按作者的公式重写：

| 项目 | 作者 | 本机 |
|---|---|---|
| 框架 | PyTorch | numpy（手写前向/反向传播 + Adam） |
| 数据 | Kaggle digit-recognizer，28×28 | Hugging Face `ylecun/mnist` 训练集 60,000 张（MIT），2×2 平均缩小到 14×14 |
| 日程 | T=1000，β 1e-4→0.02 线性 | 相同 |
| 网络 | U-Net + 自注意力 | 残差 MLP + 正弦时间嵌入（作者第 14 格同款） |
| 训练 | 50 轮，Adam lr 1e-4，batch 128 | 40,000 步，Adam lr 1e-3 余弦衰减，batch 256 |
| 采样 | DDPM 1000 步 | 相同公式（作者第 10 格） |
| 评估 | FID（256 张）= 34.34 | 未算 FID（需要 PyTorch 版 Inception）；改为最近邻距离检查是否抄训练集 |

## 环境

```
python -m venv venv
venv\Scripts\python -m pip install numpy pillow pyarrow huggingface_hub
```
numpy 2.5.4，CPU 训练，未用 GPU。

## 命令

```
# 1. 下载 MNIST（Hugging Face 公开数据集）
python -I -c "from huggingface_hub import hf_hub_download as d; print(d('ylecun/mnist','mnist/train-00000-of-00001.parquet',repo_type='dataset',local_dir='data'))"
# 2. 主模型（页面里的绿色框图片）
python -I -X utf8 ddpm_numpy.py --data data/mnist/train-00000-of-00001.parquet --out runs/main --steps 40000 --tag main
# 3. 浏览器版小模型（实验 C，导出 float16 权重）
python -I -X utf8 ddpm_numpy.py --data data/mnist/train-00000-of-00001.parquet --out runs/web --steps 40000 --hidden 384 --layers 2 --tag web --export
# 4. 把结果写进网页、复制图片
python -I -X utf8 embed_page.py runs D:\教材\Kaggle_Cases
```

## 结果与耗时（2026-11-12）

| 运行 | 参数量 | 训练耗时 | 最终损失 | 生成 64 张×1000 步 |
|---|---|---|---|---|
| main（H=512, L=3） | 1,120,452 | 1206 秒（约 20 分钟） | 0.065 | 1.3 秒 |
| web（H=384, L=2） | 520,516 | 434 秒（约 7 分钟） | 0.081 | 1.2 秒 |
| 试过的 H=160, L=2 | 145,316 | 186 秒 | 0.235 | 生成图全是黑白斑块，容量不够，弃用 |

两次运行同时跑在 CPU 上，耗时互有影响。随机种子 0（训练）、1（生成）。
生成图到训练集最近邻的平均像素距离 4.05，真图之间约 3.89：没有照抄训练图。

文件：`run_main.json`、`run_web.json`（含损失曲线）、`weights_web.f16` + `weights_web.json`（实验 C 的权重，页面里内嵌了同一份 base64）。

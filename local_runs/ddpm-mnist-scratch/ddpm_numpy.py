"""
用纯 numpy 从零训练一个 DDPM（去噪扩散概率模型），在 MNIST 手写数字（缩小到 14x14）上
真跑出“加噪 -> 训练去噪网络 -> 从纯噪声一步步生成数字”的全过程。

对应 Kaggle notebook: vikramsandu/ddpm-from-scratch-in-pytorch
  - 作者：PyTorch + U-Net，28x28，T=1000，beta 1e-4 ~ 0.02 线性，Adam lr=1e-4，50 轮
  - 这里：numpy + 带时间嵌入的残差 MLP（本机 PyTorch 被 Windows 智能应用控制拦截，无法导入），
          14x14，T=1000，beta 1e-4 ~ 0.02 线性（与作者相同），Adam，CPU 训练

用法（在本目录外运行，-I 隔离模式）：
  python -I -X utf8 ddpm_numpy.py --data <mnist train parquet> --out <输出目录> [--steps 30000] [--hidden 512] [--layers 3] [--seed 0]
"""
import argparse, io, json, os, time
import numpy as np


# ---------------- 数据 ----------------
def load_mnist(parquet_path, size):
    import pyarrow.parquet as pq
    from PIL import Image
    tb = pq.read_table(parquet_path).to_pydict()
    imgs = tb["image"]
    X = np.empty((len(imgs), 28, 28), np.float32)
    for i, d in enumerate(imgs):
        X[i] = np.asarray(Image.open(io.BytesIO(d["bytes"])), np.float32) / 255.0
    if size == 14:
        X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))  # 2x2 平均池化：28 -> 14
    elif size != 28:
        raise ValueError("size 只支持 14 或 28")
    X = X * 2 - 1  # [0,1] -> [-1,1]，与作者第 30 格相同
    return X.reshape(len(X), -1), np.array(tb["label"])


# ---------------- 扩散日程（与作者第 7、10 格相同） ----------------
def schedule(T=1000, b0=1e-4, b1=0.02):
    betas = np.linspace(b0, b1, T, dtype=np.float64)
    alphas = 1 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar


def time_embedding(t, dim):
    """作者第 14 格的正弦时间嵌入：前一半 sin，后一半 cos。"""
    factor = 10000 ** (2 * np.arange(dim // 2, dtype=np.float32) / dim)
    e = t[:, None].astype(np.float32) / factor[None, :]
    return np.concatenate([np.sin(e), np.cos(e)], axis=1)


# ---------------- 噪声预测网络：带时间嵌入的残差 MLP ----------------
def silu(x):
    s = 1 / (1 + np.exp(-x))
    return x * s, s


class Net:
    def __init__(self, D, H, L, E, rng):
        g = lambda i, o: (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)
        self.D, self.H, self.L, self.E = D, H, L, E
        self.p = {"Win": g(D + E, H), "bin": np.zeros(H, np.float32)}
        for k in range(L):
            self.p[f"W{k}"] = g(H, H) * 0.5
            self.p[f"U{k}"] = g(E, H) * 0.5
            self.p[f"b{k}"] = np.zeros(H, np.float32)
        self.p["Wout"] = (rng.standard_normal((H, D)) * 0.01).astype(np.float32)
        self.p["bout"] = np.zeros(D, np.float32)

    def forward(self, x, te, keep=False):
        p = self.p
        z0 = np.concatenate([x, te], 1)
        a0 = z0 @ p["Win"] + p["bin"]
        h, s0 = silu(a0)
        cache = [(z0, a0, s0)]
        for k in range(self.L):
            a = h @ p[f"W{k}"] + te @ p[f"U{k}"] + p[f"b{k}"]
            f, s = silu(a)
            cache.append((h, a, s))
            h = h + f
        out = h @ p["Wout"] + p["bout"]
        if keep:
            self.cache = (cache, h, te)
        return out

    def backward(self, dout):
        p, g = self.p, {}
        cache, h, te = self.cache
        g["Wout"] = h.T @ dout
        g["bout"] = dout.sum(0)
        dh = dout @ p["Wout"].T
        for k in reversed(range(self.L)):
            hin, a, s = cache[k + 1]
            da = dh * (s * (1 + a * (1 - s)))  # d silu
            g[f"W{k}"] = hin.T @ da
            g[f"U{k}"] = te.T @ da
            g[f"b{k}"] = da.sum(0)
            dh = dh + da @ p[f"W{k}"].T
        z0, a0, s0 = cache[0]
        da0 = dh * (s0 * (1 + a0 * (1 - s0)))
        g["Win"] = z0.T @ da0
        g["bin"] = da0.sum(0)
        return g

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in params:
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


# ---------------- 采样（与作者第 10、36 格相同的 DDPM 反向一步） ----------------
def sample(net, n, D, E, betas, alphas, abar, rng, snap_ts=()):
    T = len(betas)
    xt = rng.standard_normal((n, D)).astype(np.float32)
    snaps = {}
    for t in reversed(range(T)):
        te = time_embedding(np.full(n, t), E)
        eps = net.forward(xt, te)
        x0 = np.clip((xt - np.sqrt(1 - abar[t]) * eps) / np.sqrt(abar[t]), -1, 1)
        mean = (xt - (1 - alphas[t]) / np.sqrt(1 - abar[t]) * eps) / np.sqrt(alphas[t])
        if t > 0:
            var = (1 - abar[t - 1]) / (1 - abar[t]) * betas[t]
            xt = (mean + np.sqrt(var) * rng.standard_normal(xt.shape)).astype(np.float32)
        else:
            xt = mean.astype(np.float32)
        if t in snap_ts:
            snaps[t] = (xt.copy(), x0.copy())
    return np.clip(xt, -1, 1), snaps


def save_grid(X, side, path, cols, scale=4, pad=2):
    from PIL import Image
    n = len(X)
    rows = (n + cols - 1) // cols
    W = cols * (side * scale + pad) + pad
    Hh = rows * (side * scale + pad) + pad
    canvas = np.full((Hh, W), 255, np.uint8)
    for i, x in enumerate(X):
        im = ((np.clip(x.reshape(side, side), -1, 1) + 1) * 127.5).astype(np.uint8)
        im = np.kron(im, np.ones((scale, scale), np.uint8))
        r, c = divmod(i, cols)
        y0, x0 = pad + r * (side * scale + pad), pad + c * (side * scale + pad)
        canvas[y0:y0 + side * scale, x0:x0 + side * scale] = im
    Image.fromarray(canvas).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--temb", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_gen", type=int, default=64)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--export", action="store_true", help="导出 float16 权重（给浏览器用）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    X, y = load_mnist(a.data, a.size)
    N, D = X.shape
    side = a.size
    print(f"数据：{N} 张，每张 {side}x{side} = {D} 像素，读入 {time.time()-t0:.1f}s", flush=True)

    betas, alphas, abar = schedule()
    T = len(betas)
    sab, s1ab = np.sqrt(abar).astype(np.float32), np.sqrt(1 - abar).astype(np.float32)

    # 前向加噪示意图：同一张图在不同 t 下（作者第 7 格的 add_noise）
    demo_ts = [0, 50, 100, 200, 300, 500, 700, 999]
    x_demo = X[:4]
    rows = []
    for i in range(4):
        e = rng.standard_normal(D).astype(np.float32)
        for t in demo_ts:
            rows.append(sab[t] * x_demo[i] + s1ab[t] * e)
    save_grid(np.array(rows), side, os.path.join(a.out, f"forward_{a.tag}.png"), cols=len(demo_ts), scale=6)

    net = Net(D, a.hidden, a.layers, a.temb, rng)
    opt = Adam(net.p, a.lr)
    print(f"网络参数量：{net.n_params():,}", flush=True)
    losses = []
    t1 = time.time()
    run = 0.0
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, N, a.batch)
        x0 = X[idx]
        t = rng.integers(0, T, a.batch)
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        xt = sab[t, None] * x0 + s1ab[t, None] * eps          # 作者第 33 格：dfp.add_noise
        te = time_embedding(t, a.temb)
        pred = net.forward(xt, te, keep=True)                 # 作者第 33 格：model(noisy_imgs, t)
        diff = pred - eps
        loss = float((diff ** 2).mean())                      # MSE(noise_pred, noise)
        grads = net.backward(2 * diff / diff.size)
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * step / a.steps))  # 余弦衰减（作者用固定 lr=1e-4）
        opt.step(net.p, grads, lr)
        run = loss if step == 1 else 0.99 * run + 0.01 * loss
        if step % 100 == 0:
            losses.append([step, round(run, 5)])
        if step % 2000 == 0 or step == 1:
            print(f"step {step:6d}  loss(滑动) {run:.4f}  已用 {time.time()-t1:.0f}s", flush=True)
    train_sec = time.time() - t1

    t2 = time.time()
    snap_ts = (999, 800, 600, 400, 300, 200, 100, 50, 0)
    gen, snaps = sample(net, a.n_gen, D, a.temb, betas, alphas, abar, np.random.default_rng(a.seed + 1), snap_ts)
    gen_sec = time.time() - t2
    save_grid(gen, side, os.path.join(a.out, f"samples_{a.tag}.png"), cols=8, scale=5)
    # 反向过程快照：8 个样本 x 9 个时刻
    traj = []
    for i in range(8):
        for t in snap_ts:
            traj.append(snaps[t][0][i])
    save_grid(np.array(traj), side, os.path.join(a.out, f"reverse_{a.tag}.png"), cols=len(snap_ts), scale=6)

    # 记忆检查：每张生成图与训练集最近邻的距离（像素 L2），与训练集内部的最近邻距离对比
    sub = X[rng.choice(N, 5000, replace=False)]
    def nn_dist(Q):
        d = (Q ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None] - 2 * Q @ sub.T
        d.sort(1)
        return d
    dg = np.sqrt(np.maximum(nn_dist(gen)[:, 0], 0))
    dr = np.sqrt(np.maximum(nn_dist(X[:64] + 0)[:, 1], 0))  # 训练图自身距离为 0，取第二近

    info = {
        "tag": a.tag, "image_size": side, "n_train": int(N), "steps": a.steps, "batch": a.batch,
        "hidden": a.hidden, "layers": a.layers, "temb": a.temb, "params": net.n_params(),
        "T": T, "beta": [1e-4, 0.02], "lr": a.lr, "seed": a.seed,
        "train_seconds": round(train_sec, 1), "sample_seconds_for_n_gen": round(gen_sec, 1), "n_gen": a.n_gen,
        "final_loss": losses[-1][1] if losses else None, "loss_curve": losses,
        "nn_dist_generated_mean": round(float(dg.mean()), 3), "nn_dist_real_mean": round(float(dr.mean()), 3),
        "numpy": np.__version__,
    }
    json.dump(info, open(os.path.join(a.out, f"run_{a.tag}.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "loss_curve"}, ensure_ascii=False), flush=True)

    if a.export:
        order = ["Win", "bin"] + sum([[f"W{k}", f"U{k}", f"b{k}"] for k in range(a.layers)], []) + ["Wout", "bout"]
        flat = np.concatenate([net.p[k].ravel() for k in order]).astype(np.float16)
        flat.tofile(os.path.join(a.out, f"weights_{a.tag}.f16"))
        json.dump({"order": order, "shapes": {k: list(net.p[k].shape) for k in order}, "D": D, "H": a.hidden,
                   "L": a.layers, "E": a.temb, "side": side},
                  open(os.path.join(a.out, f"weights_{a.tag}.json"), "w"), indent=0)


if __name__ == "__main__":
    main()

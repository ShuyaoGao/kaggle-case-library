"""
在本机（Windows 11 + RTX 5090 Laptop）用 ONNX Runtime + DirectML 跑 Stable Diffusion，
不需要 PyTorch（本机开着 Windows 智能应用控制，PyTorch 的 DLL 被拦截，导入不了）。

对应 Kaggle notebook: burhanuddinlatsaheb/text-to-image-generation-stable-diffusion
作者的手写流水线（第 19–34 格）= 分词 -> 文本编码 -> 随机潜变量 -> 调度器循环（UNet 预测噪声 + CFG）-> VAE 解码。
这里用 numpy 把同样的步骤再写一遍，只把三个神经网络（文本编码器、UNet、VAE 解码器）交给 ONNX Runtime。

两种模式：
  --model turbo : stabilityai/sd-turbo 的 ONNX 导出（Hugging Face: Heliosoph/sd-turbo-onnx），
                  Euler 调度器 + trailing 时间步，1–4 步，不用 CFG（guidance 0）
  --model sd15  : Stable Diffusion v1.5 的 ONNX 导出（Hugging Face: onnx-community/stable-diffusion-v1-5-ONNX），
                  与作者相同的 K-LMS 调度器 + CFG 7.5（作者用的是 v1.4，v1.5 是它的后续训练版本）

用法：
  python -I -X utf8 sd_onnx_dml.py --model turbo --model_dir <模型目录> --prompt "..." --steps 1 --seed 48 --out out.png
  可选：--guidance 7.5  --snap_dir <目录>（保存每一步的“当前猜测的成品图”）
"""
import argparse, json, os, time
import numpy as np


def make_session(path, provider):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    provs = ["DmlExecutionProvider", "CPUExecutionProvider"] if provider == "dml" else ["CPUExecutionProvider"]
    if provider == "dml":
        so.enable_mem_pattern = False  # DirectML 要求
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, sess_options=so, providers=provs)


def in_type(sess, name):
    for i in sess.get_inputs():
        if i.name == name:
            return np.float16 if "float16" in i.type else (np.int64 if "int64" in i.type else (np.int32 if "int32" in i.type else np.float32))
    raise KeyError(name)


# ---------- 调度器（numpy 实现，公式同 diffusers） ----------
def base_sigmas(beta_start=0.00085, beta_end=0.012, T=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=np.float64) ** 2  # scaled_linear
    abar = np.cumprod(1 - betas)
    return np.sqrt((1 - abar) / abar)  # 每个训练时间步对应的噪声强度 sigma


def timesteps_for(n, spacing, T=1000, steps_offset=1):
    if spacing == "trailing":  # SD-Turbo
        return np.round(np.arange(T, 0, -T / n)).astype(np.float64) - 1
    # "linspace"：作者 diffusers 0.3.0 的 LMSDiscreteScheduler 用的就是这种
    return np.linspace(0, T - 1, n, dtype=np.float64)[::-1].copy()


def interp_sigmas(ts, all_sig):
    s = np.interp(ts, np.arange(len(all_sig)), all_sig)
    return np.concatenate([s, [0.0]])


def lms_coeff(order, t, cur, sigmas):
    """K-LMS 的线性多步系数：对拉格朗日基函数在 [sigma_t, sigma_{t+1}] 上积分（数值积分代替 scipy.quad）。"""
    def lms_derivative(tau):
        prod = 1.0
        for k in range(order):
            if cur == k:
                continue
            prod *= (tau - sigmas[t - k]) / (sigmas[t - cur] - sigmas[t - k])
        return prod
    a, b = sigmas[t], sigmas[t + 1]
    xs = np.linspace(a, b, 2001)
    ys = np.array([lms_derivative(x) for x in xs])
    return float(np.trapezoid(ys, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["turbo", "sd15"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snap_dir", default=None)
    ap.add_argument("--snap_every", type=int, default=1)
    ap.add_argument("--provider", default="dml", choices=["dml", "cpu"])
    a = ap.parse_args()
    if a.guidance is None:
        a.guidance = 0.0 if a.model == "turbo" else 7.5
    use_cfg = a.guidance > 1.0
    md = a.model_dir
    log = {"model": a.model, "prompt": a.prompt, "negative": a.negative, "steps": a.steps, "guidance": a.guidance,
           "seed": a.seed, "size": a.size, "provider": a.provider}

    # 1) 分词（作者第 26 格 tokenizer(...)）：CLIP BPE 分词器，补齐到 77 个位置
    t0 = time.time()
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(os.path.join(md, "tokenizer"))
    ids = tok([a.prompt], padding="max_length", max_length=tok.model_max_length, truncation=True).input_ids
    uids = tok([a.negative], padding="max_length", max_length=tok.model_max_length, truncation=True).input_ids
    n_tok = len(tok(a.prompt).input_ids)
    log["prompt_tokens"] = n_tok
    log["truncated"] = n_tok > tok.model_max_length

    # 2) 文本编码器（作者第 26 格 text_encoder(...)）
    te = make_session(os.path.join(md, "text_encoder", "model.onnx"), a.provider)
    te_in = te.get_inputs()[0]
    dt_ids = in_type(te, te_in.name)
    emb = te.run(None, {te_in.name: np.array(ids, dt_ids)})[0]
    uemb = te.run(None, {te_in.name: np.array(uids, dt_ids)})[0]
    del te
    log["text_embedding_shape"] = list(emb.shape)

    # 3) 随机潜变量（作者第 27 格 torch.randn(...)）：1 x 4 x 64 x 64
    rng = np.random.default_rng(a.seed)
    h = w = a.size // 8
    latents = rng.standard_normal((1, 4, h, w)).astype(np.float32)

    # 4) 调度器
    all_sig = base_sigmas()
    spacing = "trailing" if a.model == "turbo" else "linspace"
    ts = timesteps_for(a.steps, spacing)
    sig = interp_sigmas(ts, all_sig)
    if spacing == "trailing":
        latents = latents * np.sqrt(sig[0] ** 2 + 1)  # Euler + trailing 的 init_noise_sigma
    else:
        latents = latents * sig[0]                      # 作者第 29 格 latents * scheduler.sigmas[0]
    log["timesteps"] = [int(x) for x in ts]
    log["sigmas"] = [round(float(x), 4) for x in sig]

    unet = make_session(os.path.join(md, "unet", "model.onnx"), a.provider)
    names = [i.name for i in unet.get_inputs()]
    dt_lat = in_type(unet, "sample")
    dt_t = in_type(unet, "timestep")
    t_shape = [i.shape for i in unet.get_inputs() if i.name == "timestep"][0]
    vae = None
    if a.snap_dir:
        os.makedirs(a.snap_dir, exist_ok=True)
        vae = make_session(os.path.join(md, "vae_decoder", "model.onnx"), a.provider)
    t_load = time.time() - t0

    ctx = np.concatenate([uemb, emb]) if use_cfg else emb
    snap_set = set(range(1, a.steps + 1)) if a.snap_every <= 1 else set(list(range(a.snap_every, a.steps + 1, a.snap_every)) + [1, a.steps])
    derivs = []
    t1 = time.time()
    for i, t in enumerate(ts):
        s = sig[i]
        x_in = np.concatenate([latents] * 2) if use_cfg else latents
        x_in = x_in / np.sqrt(s ** 2 + 1)                 # 作者第 30 格 latent_model_input / ((sigma**2 + 1) ** 0.5)
        tt = np.array([t] * x_in.shape[0] if (t_shape and len(t_shape) == 1) else t, dtype=dt_t)
        feed = {"sample": x_in.astype(dt_lat), "timestep": tt, "encoder_hidden_states": ctx.astype(dt_lat)}
        eps = unet.run(None, {k: v for k, v in feed.items() if k in names})[0].astype(np.float32)
        if use_cfg:                                       # 作者第 30 格：无条件 + 7.5 x (有条件 - 无条件)
            e_u, e_c = eps[:1], eps[1:]
            eps = e_u + a.guidance * (e_c - e_u)
        x0_pred = latents - s * eps                       # 当前“猜测的成品”
        if a.model == "turbo":                            # Euler 一步
            latents = latents + (sig[i + 1] - s) * eps
        else:                                             # K-LMS（作者第 22、30 格），4 阶
            derivs.append(eps)
            if len(derivs) > 4:
                derivs.pop(0)
            order = min(i + 1, 4)
            coeffs = [lms_coeff(order, i, k, sig) for k in range(order)]
            latents = latents + sum(c * d for c, d in zip(coeffs, reversed(derivs)))
        if vae is not None and (i + 1) in snap_set:
            vn = vae.get_inputs()[0].name
            img = vae.run(None, {vn: (x0_pred / 0.18215).astype(in_type(vae, vn))})[0]
            save(img, os.path.join(a.snap_dir, f"step{i+1:03d}_x0.png"))       # 这一步“猜测的成品”
            xt_vis = latents / np.sqrt(sig[i + 1] ** 2 + 1)                      # 这一步之后的带噪潜变量（缩放到正常幅度）
            img = vae.run(None, {vn: (xt_vis / 0.18215).astype(in_type(vae, vn))})[0]
            save(img, os.path.join(a.snap_dir, f"step{i+1:03d}_xt.png"))
    t_loop = time.time() - t1

    # 5) VAE 解码（作者第 32 格 latents = 1 / 0.18215 * latents; vae.decode）
    t2 = time.time()
    if vae is None:
        vae = make_session(os.path.join(md, "vae_decoder", "model.onnx"), a.provider)
    vin = vae.get_inputs()[0].name
    img = vae.run(None, {vin: (latents / 0.18215).astype(in_type(vae, vin))})[0]
    t_dec = time.time() - t2
    save(img, a.out)
    # 潜空间 4 个通道的可视化（给页面讲“潜空间”用）
    lat_png = os.path.splitext(a.out)[0] + "_latent.png"
    save_latent(latents, lat_png)
    log.update({"seconds_load": round(t_load, 2), "seconds_denoise_loop": round(t_loop, 2), "seconds_vae": round(t_dec, 2),
                "seconds_total": round(time.time() - t0, 2), "out": os.path.basename(a.out)})
    json.dump(log, open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False))


def save(img, path):
    from PIL import Image
    x = np.clip(img[0].astype(np.float32) / 2 + 0.5, 0, 1)   # 作者第 34 格 (image / 2 + 0.5).clamp(0, 1)
    Image.fromarray((x.transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(path)


def save_latent(lat, path):
    from PIL import Image
    L = lat[0]
    tiles = []
    for c in range(4):
        v = L[c]
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        tiles.append(np.kron((v * 255).astype(np.uint8), np.ones((2, 2), np.uint8)))
    Image.fromarray(np.concatenate(tiles, 1)).save(path)


if __name__ == "__main__":
    main()

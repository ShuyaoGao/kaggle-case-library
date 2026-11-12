# Kaggle 案例库：255 个机器学习实战案例的中文拆解

在线浏览：**https://shuyaogao.github.io/kaggle-case-library/**

把 Kaggle 上的高票公开 notebook 拆解成中文的动态教学网页，给零基础的听众讲清楚：用了什么数据、要解决什么问题、每一步用了什么方法、做到了什么。

- **255 个案例**，覆盖 11 个应用领域：金融商业 40 · 文化娱乐与互联网 40 · 医学健康 38 · 通用方法 22 · 教育社会 21 · 交通城市 17 · 生物农业 17 · 工程制造 16 · 网络安全 16 · 能源环境 15 · 心理与行为 13
- **六类问题**：机器学习 121 · 深度学习 117 · 强化学习 12 · 数据挖掘 32 · 优化与运筹 15 · 统计分析 25（一个案例可以同时属于几类）
- 每个案例：数据长什么样 → 按原 notebook 顺序逐章拆解（动画 + 中文讲解 + 原作者全部代码）→ 在浏览器里现场计算的交互实验 → 结论与讨论题
- 每个案例配一份 **Word 讲稿**（`讲稿/` 文件夹），和网页位置一一对应，用初中生能听懂的类比讲解

## 入口

| 页面 | 内容 |
|---|---|
| [showcase.html](https://shuyaogao.github.io/kaggle-case-library/showcase.html) | 总览：六类问题、方法地图、领域 × 方法矩阵、按领域展开 |
| [routes.html](https://shuyaogao.github.io/kaggle-case-library/routes.html) | 30 条学习路线：按难度、领域、问题类型、数据类型、专题 |
| [index.html](https://shuyaogao.github.io/kaggle-case-library/) | 卡片目录：按领域、范式、方法筛选和搜索 |
| [智能搜索](https://shuyaogao.github.io/case-rag-search/) | 用自然语言搜案例，并一步步演示 RAG 向量检索的原理（独立项目） |

下载本仓库后双击 `index.html` 也能离线使用，不需要联网或服务器。

## 诚实说明

- 页面只展示原作者 notebook 的源码；Kaggle API 拉下的 notebook 不带运行输出，所以没有作者在文字里写明的分数一律不列。
- 动画和实验里的数据是浏览器内模拟，页面上都有标注；不编造真实样例行。
- 原 notebook 里发现的错误（数据泄漏、文字与代码不一致等）如实写在页面和讨论题里，不改原文。
- 两个扩散模型案例附有在本机真跑的结果和可复现脚本（`local_runs/`），与作者结果分开标注。

## 版权

- 各案例的原 notebook 版权归原作者所有，每页都标明作者并链接到 Kaggle 原文。Kaggle 公开 notebook 默认以 Apache 2.0 许可发布。
- 中文讲解、动画、交互实验和讲稿为本库原创，供教学使用。
- 公式排版使用 [KaTeX](https://katex.org/)（MIT 许可），放在 `assets/katex/`。

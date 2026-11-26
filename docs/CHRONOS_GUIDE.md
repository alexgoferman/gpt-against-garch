# Chronos: Zero-Shot and Fine-Tuned (owner: Maggie)

[English](#english) · [中文](#中文版)

A visual overview of the whole pipeline is in [`pipeline_map.html`](pipeline_map.html).

<a id="english"></a>
## English

### Files
| File | Purpose |
|---|---|
| `src/data_utils.py` | Reads the output of `preprocess.py` (value- or equal-weighted) and converts % to decimal returns. The split dates (≤2010 / 2011–15 / ≥2016) come from `preprocess.SPLITS`. |
| `src/chronos_zero_shot.py` | Rolling one-day-ahead forecasts. For each day, the model sees only the previous 512 days and draws 100 samples. Pass `--model` to run a fine-tuned checkpoint with the same script. |
| `src/prepare_finetune_data.py` | Writes the training and validation series as Arrow files, the format the Chronos training script reads. |
| `configs/chronos-t5-mini-finetune.yaml` | Fine-tuning config. Starts from the pretrained chronos-t5-mini weights. |

### Setup (Jetstream2 A100)
```bash
git clone https://github.com/amazon-science/chronos-forecasting
cd chronos-forecasting && pip install -e ".[training]" && cd ..   # includes the training script's dependencies
pip install "gluonts[arrow]" pandas                               # ArrowWriter for prepare_finetune_data.py
pip freeze > requirements.txt                                     # the project PDF asks us to pin versions
```
- The clone gives us `scripts/training/train.py`, the official training script. `pip install chronos-forecasting` alone does not include it.
- To try only the zero-shot model on a laptop, `pip install chronos-forecasting` is enough.

Put the Ken French CSV at `data/raw/49_Industry_Portfolios_Daily.csv` and keep that copy fixed. All scripts load data through `data_utils.load_french49()`, which reads `data/processed/returns_{vw,ew}.csv` produced by `src/preprocess.py` (it runs preprocess automatically if those files are missing). No `--csv` argument is needed.

### 1. Zero-shot (do this first, fastest)
```bash
# 1) Smoke test: only the first 5 test days, to check that everything runs (about a minute)
python src/chronos_zero_shot.py --period test --max_days 5
# 2) Full run: ~2,700 test days × 49 industries ≈ 130,000 forecasts
python src/chronos_zero_shot.py --period test
python src/chronos_zero_shot.py --table equal --period test      # equal-weighted (optional)
```
Output: `results/forecasts/chronos_zs_value_test.npz`, where `samples` has shape `[n_days, 49, 100]`.

**Shared format for teammates:** please save the GARCH and random-walk forecasts in the same format (same dates, same industry order, `[n_days, 49, n_samples]`). If GARCH gives an analytic distribution, draw 100 samples from its Student-t. Then `metrics.py` can compute CRPS, coverage and QLIKE for all four models with the same code.

### 2. Fine-tuned
```bash
# 1) Prepare data. Starting in 1970 is suggested; explain why in the report
python src/prepare_finetune_data.py --start 1970-01-01
# 2) Smoke test: set max_steps to 200 in the yaml and check that the loss goes down
python chronos-forecasting/scripts/training/train.py --config configs/chronos-t5-mini-finetune.yaml
# 3) Learning-rate sweep: one run each at 1e-5, 3e-5, 1e-4 (copy the yaml, change learning_rate and output_dir)
# 4) Score every checkpoint on the validation period and pick the one with the lowest CRPS
python src/chronos_zero_shot.py --model checkpoints/run-1/checkpoint-3000 --name chronos_ft_lr3e-5_s3000 --period val
# 5) Run the test period only once, with the selected checkpoint
python src/chronos_zero_shot.py --model <best_ckpt> --name chronos_ft --period test
```
Note: the yaml field names follow the official `scripts/training/configs/chronos-t5-mini.yaml`. Compare against it after cloning, in case a newer version renamed anything.

### What to write in README section 4.4 (the 9 questions in the project PDF)
- **Model architecture** (Q4): T5 encoder-decoder; chronos-t5-mini has about 20M parameters and a 4,096-token vocabulary. Read width, number of layers and heads from the checkpoint's `config.json`. State that we fine-tune a public pretrained model rather than train from scratch. A Chinchilla-style argument (N ≈ D/20) helps: our training data is only about 0.5–1M tokens, which could support only a very small model trained from scratch.
- **Scaling and tokenization** (Q3): each input window is divided by its mean absolute value, then the scaled values are binned uniformly into 4,094 bins on [-15, 15]; with PAD and EOS that makes 4,096 tokens. Give one worked example: a 1.2% return with a window mean absolute value of 0.8% scales to 1.5 and falls in bin number X. Possible discussion: returns sit near 0 with heavy tails, so most bins are never used.
- **Training objective** (Q5): cross-entropy over the next bin (a classification loss), optimized with AdamW.
- **Hyperparameter selection** (Q6): learning-rate sweep, 5% warmup, linear decay, context 512, prediction_length 8, plus settings we tried that did not help.
- **Training and validation loss** (Q8): training-loss curves for the three learning rates, plus validation CRPS for each checkpoint.
- **Checkpoint selection**: choose by CRPS on the validation period (2011–15); evaluate the test period only once to avoid information leakage.

Also record (Q7): training steps, batch size (32), total tokens, GPU (A100) and hours used. Log every run in one table.

---

<a id="中文版"></a>
## 中文版

### 文件
| 文件 | 作用 |
|---|---|
| `src/data_utils.py` | 读取 `preprocess.py` 的输出（市值加权/等权），% 转小数；切分日期 ≤2010 / 2011–15 / ≥2016 来自 `preprocess.SPLITS` |
| `src/chronos_zero_shot.py` | 滚动的一天后预测：每天只用此前 512 天作为输入，采样 100 个；换 `--model` 参数即可跑微调后的模型 |
| `src/prepare_finetune_data.py` | 把训练集和验证集写成 Chronos 训练脚本要求的 Arrow 格式 |
| `configs/chronos-t5-mini-finetune.yaml` | 微调配置（从 chronos-t5-mini 的预训练权重出发） |

### 环境（Jetstream2 A100）
```bash
git clone https://github.com/amazon-science/chronos-forecasting
cd chronos-forecasting && pip install -e ".[training]" && cd ..   # 包含训练脚本需要的依赖
pip install "gluonts[arrow]" pandas                               # prepare_finetune_data.py 要用 ArrowWriter
pip freeze > requirements.txt                                     # PDF 要求锁定版本
```
- 克隆仓库是为了拿到官方训练脚本 `scripts/training/train.py`。只装 `pip install chronos-forecasting` 不包含这个脚本。
- 如果只想在自己电脑上试零样本，装 `pip install chronos-forecasting` 就够了。

把 French 的 CSV 放在 `data/raw/49_Industry_Portfolios_Daily.csv`，并固定这份副本。所有脚本都通过 `data_utils.load_french49()` 读取 `src/preprocess.py` 生成的 `data/processed/returns_{vw,ew}.csv`（文件不存在时会自动运行 preprocess），因此不再需要 `--csv` 参数。

### 一、Zero-Shot（先做，最快）
```bash
# 1) 冒烟测试：只跑测试期前 5 天，确认流程能走通（约一分钟）
python src/chronos_zero_shot.py --period test --max_days 5
# 2) 正式：测试期约 2700 天 × 49 个行业 ≈ 13 万次预测
python src/chronos_zero_shot.py --period test
python src/chronos_zero_shot.py --table equal --period test      # 等权（可选）
```
输出：`results/forecasts/chronos_zs_value_test.npz`，其中 `samples` 的形状是 [天数, 49, 100]。

**和队友的接口约定**：GARCH 和随机游走也存成同样的格式（同样的日期、行业顺序、[天数, 49, 样本数]）。如果 GARCH 用解析分布，就从 t 分布里抽 100 个样本。这样 `metrics.py` 可以用同一套代码给四个模型算 CRPS、覆盖率和 QLIKE。

### 二、Fine-Tuned
```bash
# 1) 准备数据。建议从 1970 年开始，在报告里说明原因
python src/prepare_finetune_data.py --start 1970-01-01
# 2) 冒烟测试：把 yaml 里的 max_steps 改成 200，确认 loss 在下降
python chronos-forecasting/scripts/training/train.py --config configs/chronos-t5-mini-finetune.yaml
# 3) 学习率扫描：分别用 1e-5、3e-5、1e-4 各跑一次（复制 yaml，改 learning_rate 和 output_dir）
# 4) 对每个 checkpoint 在验证期打分，选验证 CRPS 最低的
python src/chronos_zero_shot.py --model checkpoints/run-1/checkpoint-3000 --name chronos_ft_lr3e-5_s3000 --period val
# 5) 只用选中的那一个 checkpoint 去跑测试期，测试期只跑一次
python src/chronos_zero_shot.py --model <best_ckpt> --name chronos_ft --period test
```
注意：yaml 里的字段名是按官方 `scripts/training/configs/chronos-t5-mini.yaml` 写的。clone 下来后对照一遍，以防版本更新改了名字。

### README 4.4 各小节怎么写（对应 PDF 的 9 个问题）
- **Model architecture**（第 4 题）：T5 编码器-解码器，chronos-t5-mini 约 2000 万参数，词表 4096 个 token。宽度、层数、头数从 checkpoint 的 `config.json` 里读出来再填。说明是从公开的预训练模型出发微调，而不是从零训练。可以用 Chinchilla 的 N≈D/20 论证：训练数据只有约 50–100 万个 token，从零训练只撑得起极小的模型
- **Scaling and tokenization**（第 3 题）：用输入窗口的平均绝对值做缩放，再均匀切分到 [-15, 15] 区间的 4094 个 bin，加上 PAD 和 EOS 共 4096 个 token。举一个例子：某天收益 1.2%，窗口平均绝对值 0.8%，缩放后是 1.5，对应第几个 bin。可以讨论：收益在 0 附近、尾部很厚，大部分 bin 用不到
- **Training objective**（第 5 题）：预测下一个 bin 的交叉熵（分类损失），优化器 AdamW
- **Hyperparameter selection**（第 6 题）：学习率扫描、warmup 5%、线性衰减、context 512、prediction_length 8，以及试过但没效果的设置
- **Training and validation loss**（第 8 题）：三个学习率的训练 loss 曲线，加上每个 checkpoint 的验证 CRPS
- **Checkpoint selection**：按验证期（2011–15）的 CRPS 选，测试期只评估一次，避免信息泄露

还要记录（第 7 题）：训练步数、batch size（32）、总 token 数、A100、用了几小时。每次运行都记到一张表里。

# Lab 4.5 — Transformer 翻译模型（CPU + DLP）

## 文件归位

云平台目录：`/opt/code_chap4/exp_4_5_transformer/`

```bash
LAB=Intelligent-Computing-Lab/lab4.5
DST=/opt/code_chap4/exp_4_5_transformer
cp $LAB/modules.py  $DST/
cp $LAB/AttModel.py $DST/
cp $LAB/eval.py     $DST/
```

## 运行（重要：先清空 output.txt）

`eval.py` 用 `'a'` 模式追加写日志，框架自带的 `output.txt` 里有参考输出，**不清空会污染 BLEU**：

```bash
cd /opt/code_chap4/exp_4_5_transformer
> output.txt        # 清空旧日志

# CPU 推理（约 7-8 分钟）
bash run_transformer_cpu.sh

# DLP 推理
bash run_transformer_mlu.sh
```

期望终端输出末尾出现 `Bleu Score = 17.xx` 与 `Transformer {CPU,MLU} PASS!`。

## 实现要点

### `modules.py`
- **embedding**：`F.embedding(inputs, lookup_table, padding_idx, None, 2, False, False)`，scale 时 `* sqrt(d)`，初始化用 `nn.init.xavier_normal_`。
- **layer_normalization**：γ/β 用 `nn.Parameter(torch.ones/zeros(features))`；除法是 `(x-mean)/(std+eps)`，eps 加在分母外侧。
- **positional_encoding**：`pos / 10000^(2i/d)` 矩阵，偶列 sin、奇列 cos，`zeros_pad=True` 时把第 0 行填零。
- **multihead_attention**：split-concat 用 `cat(chunk(Q, h, dim=2), dim=0)`；scale 除以 `sqrt(d_k) = sqrt(C/h)`；causality 分支用 `torch.tril`；softmax 沿 `dim=-1`；query mask 后 dropout 后再 `bmm` V；最后残差 + LN。
- **feedforward**：默认 `conv=False` 走 Linear 路径，`Sequential(Linear(in, 4d), ReLU)` + `Linear(4d, d)`，残差 `+= inputs`，再 LN。
- **label_smoothing**：`(1-eps)*x + eps/K`，`K = inputs.size(-1)`。

### `AttModel.py`
- Encoder：`enc_self_attention_i(self.enc, self.enc, self.enc)` → `enc_feed_forward_i(self.enc)`。
- Decoder：先 self-attn（causality=True）→ **cross-attn `(self.dec, self.enc, self.enc)`** → feed forward。**易错点**：vanilla（cross）attention 的 keys/values 必须来自 encoder。
- 末端 `logits = Linear(d, V)` → `probs = softmax(logits).view(-1, V)` → `preds = argmax(logits, -1)`。
- ffn 中间维度按论文取 4×hidden = 2048。

### `eval.py`
- 数据集解包是三元组：`for i, (x, sources, targets) in enumerate(test_loader)`。
- 模型推理是**自回归**：`for j in range(maxlen): preds_t[:, j] = model(x_, preds).preds[:, j]`。
- BLEU：`corpus_bleu(list_of_refs, hypotheses)`，过滤掉长度 ≤3 的句子。
- `qps = total_samples / time`。

## 与官方参考输出对比

仓库自带 `output.txt` 末行 `Bleu Score = 17.095544451082745` —— 这是 model_epoch_09 + IWSLT2014 测试集的标准答案。本实现的 BLEU 应该非常接近 17.10（数值层面除非 dropout 行为差异，否则应一致）。

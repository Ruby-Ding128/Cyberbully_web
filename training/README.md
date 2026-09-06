# 轻量化 BERT 多标签分类

默认模型为 `distilbert-base-uncased`，输出 10 个标签：

- 原攻击标签：`severe_toxicity`、`obscene`、`identity_attack`、`insult`、`threat`、`sexual_explicit`
- 新增标签：`age`、`ethnicity`、`gender`、`religion`

`target` 是总体毒性评分，默认不作为具体攻击类别；添加 `--include-target` 可将其作为第 11 个标签。

## 安装

```powershell
python -m pip install -r training/requirements.txt
```

## 快速试跑

先用少量数据检查环境和流程：

```powershell
python training/train_multilabel_distilbert.py --sample-size 2000 --epochs 1
```

## 正式训练

```powershell
python training/train_multilabel_distilbert.py `
  --data-path data/combined_selected_categories.csv `
  --output-dir outputs/distilbert_multilabel `
  --epochs 3 `
  --train-batch-size 16 `
  --gradient-accumulation-steps 2 `
  --max-length 256
```

显存不足时可将 `--train-batch-size` 调成 8 或 4，并相应增大梯度累积步数。

## 预测

```powershell
python training/predict_multilabel.py `
  --model-dir outputs/distilbert_multilabel/final_model `
  --text "示例英文评论"
```

## 数据处理说明

- 标签值支持 0/1 和 0～1 连续软标签。
- 对结构性缺失标签使用掩码损失，不把空值误当成 0。
- 对类别不平衡使用训练集计算的 `pos_weight`，并限制最大权重为 20。
- 按规范化文本哈希分组进行 80%/10%/10% 切分，重复文本不会跨集合。
- 训练后在验证集上为每个标签独立搜索 F1 最优阈值。

## transformers 版本问题

脚本会自动兼容 `eval_strategy`/`evaluation_strategy`，并在旧版本不支持
`warmup_ratio` 时自动换算为 `warmup_steps`。如果终端显示的版本和实际运行
版本不一致，请使用同一个 Python 检查和运行：

```powershell
python -c "import sys, transformers; print(sys.executable); print(transformers.__version__)"
python training/train_multilabel_distilbert.py --sample-size 2000 --epochs 1
```

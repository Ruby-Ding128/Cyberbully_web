"""加载训练后的多标签 DistilBERT 模型进行预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/distilbert_multilabel/final_model"))
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    thresholds = json.loads((args.model_dir / "thresholds.json").read_text(encoding="utf-8"))

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        return_token_type_ids=False,
    )
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(**inputs).logits)[0].cpu().tolist()

    results = []
    for index, probability in enumerate(probabilities):
        label = model.config.id2label[index]
        threshold = thresholds.get(label, 0.5)
        results.append(
            {
                "label": label,
                "probability": round(probability, 6),
                "threshold": threshold,
                "predicted": probability >= threshold,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

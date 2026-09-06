from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .schemas import ClassificationResult, LabelScore


LABEL_ZH = {
    "severe_toxicity": "严重毒性攻击",
    "obscene": "粗俗或淫秽内容",
    "identity_attack": "身份攻击",
    "insult": "侮辱",
    "threat": "威胁",
    "sexual_explicit": "露骨色情内容",
    "age": "年龄歧视或年龄欺凌",
    "ethnicity": "种族或族裔欺凌",
    "gender": "性别相关欺凌",
    "religion": "宗教相关欺凌",
}


class CyberbullyingClassifier:
    def __init__(self, model_dir: Path, max_length: int = 256):
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在：{model_dir}")
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        threshold_path = model_dir / "thresholds.json"
        self.thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))

    @torch.inference_mode()
    def predict(self, text: str) -> ClassificationResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        ).to(self.device)
        probabilities = torch.sigmoid(self.model(**inputs).logits)[0].cpu().tolist()

        scores: list[LabelScore] = []
        for index, probability in enumerate(probabilities):
            label = self.model.config.id2label[index]
            threshold = float(self.thresholds.get(label, 0.5))
            scores.append(
                LabelScore(
                    label=label,
                    label_zh=LABEL_ZH.get(label, label),
                    probability=float(probability),
                    threshold=threshold,
                    predicted=probability >= threshold,
                )
            )

        detected = [score for score in scores if score.predicted]
        detected.sort(key=lambda item: item.probability - item.threshold, reverse=True)
        primary = detected[0] if detected else None
        return ClassificationResult(
            is_cyberbullying=bool(detected),
            primary_type=primary.label if primary else None,
            primary_type_zh=primary.label_zh if primary else None,
            detected_types=[item.label for item in detected],
            detected_types_zh=[item.label_zh for item in detected],
            scores=sorted(scores, key=lambda item: item.probability, reverse=True),
        )

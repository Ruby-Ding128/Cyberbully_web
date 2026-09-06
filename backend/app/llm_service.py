from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from .schemas import ClassificationResult, LLMResult
from .visualization import locate_phrases


SYSTEM_PROMPT = """你是面向青少年网络安全场景的内容分析助手。
你的任务是找出网络欺凌文本中的关键原文片段，并生成温和、有效、不会责怪受害者的干预提醒。
必须只返回 JSON，不要返回 Markdown。keywords.text 必须是输入文本中逐字存在的连续片段，不得改写。
不要重复或扩写辱骂内容。若涉及明确的人身威胁，提醒用户保存证据、联系可信任成年人或平台，并在有现实危险时联系当地紧急服务。
输出结构：
{"keywords":[{"text":"原文片段","category":"标签英文名","severity":"low|medium|high","explanation":"简短中文解释"}],"intervention":"2至4句适合青少年的中文提醒"}
"""


class LLMService:
    def __init__(self, api_key: str | None, model: str, base_url: str | None, timeout: float):
        self.model = model
        self.configured = bool(api_key)
        self.client = (
            AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            if api_key
            else None
        )

    async def analyze(
        self,
        text: str,
        classification: ClassificationResult,
        response_language: str,
    ) -> LLMResult:
        if not classification.is_cyberbullying:
            return LLMResult(
                provider="skipped",
                highlights=[],
                intervention="未检测到明显的网络欺凌内容。仍建议保持尊重、友善的交流方式。",
            )
        if self.client is None:
            return self._fallback(text, classification)

        payload = {
            "text": text,
            "detected_types": classification.detected_types,
            "detected_types_zh": classification.detected_types_zh,
            "response_language": response_language,
        }
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            highlights = locate_phrases(text, parsed.get("keywords", []))
            intervention = str(parsed.get("intervention", "")).strip()
            if not intervention:
                intervention = self._fallback(text, classification).intervention
            return LLMResult(provider="llm", highlights=highlights, intervention=intervention)
        except Exception:
            # 不向客户端泄漏密钥、上游 URL 或提供商错误详情。
            return self._fallback(text, classification)

    @staticmethod
    def _fallback(text: str, classification: ClassificationResult) -> LLMResult:
        # 无 LLM 配置时只标记保守的明显风险短语，不假装是 LLM 结果。
        patterns = {
            "threat": [r"\b(?:kill|hurt|attack|die)\b", r"杀死|弄死|打死|威胁"],
            "insult": [r"\b(?:idiot|stupid|loser|ugly)\b", r"笨蛋|白痴|废物|丑八怪"],
            "obscene": [r"\b(?:fuck|bitch)\b", r"他妈的|婊子"],
        }
        raw_items = []
        for category in classification.detected_types:
            for pattern in patterns.get(category, []):
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    raw_items.append(
                        {
                            "text": match.group(0),
                            "category": category,
                            "severity": "high" if category == "threat" else "medium",
                            "explanation": "可能与检测到的网络欺凌类型相关",
                        }
                    )
        # 明确描述欺凌行为的短语应被标出，即使具体分类标签不是 insult。
        generic_patterns = [
            r"\b(?:bully|bullied|bullying)\s+(?:him|her|them|you|me)\b",
            r"\b(?:bully|bullied|bullying)\b",
            r"欺负(?:他|她|他们|你|我)?",
            r"网络欺凌",
        ]
        for pattern in generic_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw_items.append(
                    {
                        "text": match.group(0),
                        "category": classification.primary_type or "cyberbullying",
                        "severity": "medium",
                        "explanation": "该片段直接表达或描述了欺凌行为",
                    }
                )
        reminder = (
            "这段话可能会让人感到受伤或不安全。请先停下来，换成描述感受和事实的表达；"
            "如果你是被针对的人，可以保存证据并告诉可信任的成年人、老师或平台管理员。"
        )
        if "threat" in classification.detected_types:
            reminder += "如果存在现实人身危险，请立即远离风险并联系当地紧急服务。"
        return LLMResult(
            provider="fallback",
            highlights=locate_phrases(text, raw_items),
            intervention=reminder,
        )

"""P7 · C 微调分类器（RoBERTa-base）—— 前端单文件接口

    from P7_C_predict import predict
    predict("Artificial intelligence has the potential to ...")
    # -> {"prob_ai": 0.97, "label": "AI", "extra": {...}}

模型按名字从 Hugging Face 自动下载（第一次约 250 MB，之后走本地缓存）。
本文件只做加载和推理，训练代码在 notebooks/P7_C.ipynb。
检测结果是概率判断，不能单独作为认定学术不端的依据。
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "Hanxi79/p7-detector"   # Hugging Face 公开仓库
MODEL = os.environ.get("P7_C_MODEL", REPO)   # 也可以设成本地文件夹路径
MAX_LEN = 256        # 与训练时一致
THRESHOLD = 0.5

_CACHE: dict = {}


def load_model(name_or_path: str | None = None):
    """按 Hugging Face 仓库名或本地路径加载模型（带缓存）。返回 (tokenizer, model, device)。"""
    key = str(name_or_path or MODEL)
    if key not in _CACHE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(key)
        # 仓库里存的是 fp16；转成 fp32，CPU 上也能跑（不用 torch_dtype/dtype 参数，兼容新旧 transformers）
        model = AutoModelForSequenceClassification.from_pretrained(key).float()
        _CACHE[key] = (tok, model.to(device).eval(), device)
    return _CACHE[key]


@torch.no_grad()
def score_texts(texts: list[str], name_or_path: str | None = None, batch_size: int = 32) -> list[float]:
    """批量返回 AI 概率（第 1 类 = AI）。超过 MAX_LEN 个 token 的部分截断。"""
    tok, model, device = load_model(name_or_path)
    texts = [t if isinstance(t, str) else "" for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out = [0.0] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        x = tok([texts[i] for i in idx], return_tensors="pt", truncation=True,
                max_length=MAX_LEN, padding=True).to(device)
        probs = torch.softmax(model(**x).logits.float(), dim=-1)[:, 1].tolist()
        for i, p in zip(idx, probs):
            out[i] = float(p)
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def predict(text: str) -> dict:
    """组内统一格式：{"prob_ai": 0–1, "label": "AI"/"Human", "extra": {...}}

    extra:
      truncated  原文是否超过 256 个 token 被截断
      sentences  [{"text", "prob_ai"}] 逐句 AI 概率（短句不可靠，仅供界面展示）
    """
    if not isinstance(text, str) or not text.strip():
        return {"prob_ai": 0.0, "label": "Human", "extra": {"empty": True}}
    text = text.strip()
    tok, _, _ = load_model()
    p = score_texts([text])[0]
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    sent_scores = score_texts(sents) if len(sents) > 1 else []
    return {
        "prob_ai": round(p, 4),
        "label": "AI" if p >= THRESHOLD else "Human",
        "extra": {
            "truncated": len(tok(text)["input_ids"]) > MAX_LEN,
            "sentences": [{"text": s, "prob_ai": round(q, 4)} for s, q in zip(sents, sent_scores)],
        },
    }


if __name__ == "__main__":
    import json
    demo = ("Artificial intelligence has the potential to transform education by providing personalized "
            "learning experiences. However, it is important to consider the ethical implications, including "
            "privacy concerns and the risk of over-reliance on technology.")
    print(json.dumps(predict(demo), ensure_ascii=False, indent=2))

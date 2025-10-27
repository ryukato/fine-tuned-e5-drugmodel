# 🧬 Fine-tuned E5-small for Korean Drug Product Semantic Embedding

## 📘 Model Overview
이 모델은 **[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)** 기반으로,
의약품 요약·상세 데이터(`drug_summary`, `drug_details`) 및 제품 유형 정의(`drug_type_definition`)를 활용하여
한국어 의약품 도메인에 맞게 **2단계 파인튜닝(fine-tuning)** 된 버전입니다.

### 🔹 1단계: Drug Type Semantic Alignment
- 데이터셋: `drug_type_def_list.csv`
- 목표: `"해열제" → "체온을 낮추는 약"` 과 같은 개념 매핑 학습  
- 모델 결과: `/tunning/model/fine_tuned_e5_small_drugtype`

### 🔹 2단계: Drug Product Semantic Alignment
- 데이터셋: `drug_product_similarity_train.csv` (약 3,000건 샘플)
- 목표: `"판콜에이내복액"` 같은 실제품과 `"열을 내리는 약"` 같은 질의 간 의미 매칭 강화  
- 모델 결과: `/tunning/model/fine_tuned_e5_small_drugproduct`

---

## 🧠 Use Case Example

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("ryukato/fine-tuned-e5-small-drugproduct")

query = "열을 내리는 약은?"
docs = [
    "판콜에이내복액은 해열진통제입니다.",
    "마이암부톨정은 항결핵제입니다.",
    "지르텍정은 항히스타민제입니다."
]

emb_q = model.encode(query, convert_to_tensor=True)
emb_d = model.encode(docs, convert_to_tensor=True)

scores = util.cos_sim(emb_q, emb_d)[0]
for doc, score in zip(docs, scores):
    print(f"{doc} → 유사도: {score.item():.4f}")
```

---

## ⚙️ Training Environment

| 항목 | 버전 |
|------|------|
| Python | 3.12.4 |
| torch | 2.4.1 |
| transformers | 4.44.2 |
| sentence-transformers | 3.0.1 |
| accelerate | 0.27.0 |
| pandas | 2.2.3 |

---

## 🧩 Directory Structure

```
fine-tuned-e5-small-drugmodel/
├── data/
│   └── drug_product_similarity_train.csv
├── scripts/
│   ├── train_drug_type_e5_small.py
│   ├── train_drug_product_e5_small.py
│   └── infer_example.py
├── model_cards/
│   └── README.md
└── requirements.txt
```

---

## 📅 Release Info
- Author: **@ryukato**
- Base Model: `intfloat/multilingual-e5-small`
- Fine-tuned Model: `fine_tuned_e5_small_drugproduct`
- Last Updated: **2025-10-27**

---

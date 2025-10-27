---
language: ko
license: mit
tags:
  - sentence-transformers
  - semantic-search
  - medical
  - pharmaceutical
  - korean
datasets:
  - drug_product_similarity_train
library_name: sentence-transformers
pipeline_tag: feature-extraction
base_model: intfloat/multilingual-e5-small
model_name: Yoonyoul/fine-tuned-e5-small-drugproduct
model_type: sentence-transformer
---

# 🧬 Fine-tuned E5-small for Korean Drug Product Semantic Embedding

## 📘 Model Overview
이 모델은 **[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)** 기반으로,  
의약품 요약·상세 데이터(`drug_summary`, `drug_details`) 및 제품 유형 정의(`drug_type_definition`)를 활용하여  
한국어 의약품 도메인에 맞게 **2단계 파인튜닝(fine-tuning)** 된 SentenceTransformer 모델입니다.

- GitHub Repository: [https://github.com/ryukato/fine-tuned-e5-drugmodel](https://github.com/ryukato/fine-tuned-e5-drugmodel)

---

## 🧩 Base Model Selection Rationale

이 프로젝트는 다국어 환경에서도 **의약품 명칭, 효능, 제형 등의 복잡한 의미 관계를 정확히 임베딩**하기 위해  
**E5(multilingual-E5)** 계열 모델 중 `intfloat/multilingual-e5-small`을 선택했습니다.

선정 이유는 다음과 같습니다:

1. **다국어 문장 표현력**  
   - 영어뿐 아니라 한국어, 일본어, 중국어, 독일어 등 다양한 언어에서 균형 잡힌 의미 표현 성능을 보여줍니다.  
   - 의약품 데이터는 외래어·학술용어가 혼합된 형태가 많기 때문에 multilingual encoder가 유리합니다.

2. **효율적 성능 대비 파라미터 크기 (Small Variant)**  
   - `small` 모델은 약 **33M 파라미터**로, M1/M2 맥북 등 로컬 환경에서도 안정적으로 fine-tuning 가능했습니다.  
   - FP16 또는 bfloat16 지원으로 GPU·MPS 환경에서도 효율적인 연산을 제공합니다.

3. **문장 단위 의미 검색(semantic retrieval)에 최적화**  
   - E5 모델은 “문장 단위 의미 임베딩(Sentence Embedding)”을 위해 학습되어 있어,  
     단순 질의(`"기침약"`, `"열 내리는 약"`)와 제품명(`"판콜에이"`, `"타이레놀"`) 간 의미 매칭에 뛰어난 성능을 보입니다.

4. **Sentence-Transformers와 완벽한 호환성**  
   - `SentenceTransformer` 인터페이스와 100% 호환되어, PyTorch 기반 pipeline 통합이 용이했습니다.

---

## 🔹 1단계: Drug Type Semantic Alignment
- 데이터셋: `drug_type_def_list.csv`  
- 목표: `"해열제" → "체온을 낮추는 약"` 과 같은 개념 매핑 학습  
- 모델 결과: `/tunning/model/fine_tuned_e5_small_drugtype`

### 🔹 2단계: Drug Product Semantic Alignment
- 데이터셋: `drug_product_similarity_train.csv` (약 3,000건 샘플)  
- 목표: `"판콜에이내복액"` 같은 실제 제품과 `"열을 내리는 약"` 같은 질의 간 의미 매칭 강화  
- 모델 결과: `/tunning/model/fine_tuned_e5_small_drugproduct`

---

## 🧠 Use Case Example

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Yoonyoul/fine-tuned-e5-small-drugproduct")

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
- Author: **@Yoonyoul**
- Base Model: `intfloat/multilingual-e5-small`
- Fine-tuned Model: `Yoonyoul/fine-tuned-e5-small-drugproduct`
- Repository: [https://github.com/ryukato/fine-tuned-e5-drugmodel](https://github.com/ryukato/fine-tuned-e5-drugmodel)
- Last Updated: **2025-10-27**

---

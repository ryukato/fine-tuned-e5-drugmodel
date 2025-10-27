# 🧬 Fine-tuned E5-small for Korean Drug Product Semantic Embedding

## 📘 Overview

이 저장소는 **[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)** 기반의  
한국어 의약품 의미 임베딩(Semantic Embedding) 모델을 구축하기 위한 전체 학습 파이프라인을 포함합니다.

이 프로젝트는 **의약품 유형(`drug_type_definition`) → 실제 제품(`drug_summary`, `drug_details`)** 으로 이어지는  
2단계 파인튜닝(fine-tuning) 과정을 통해, 질의("열을 내리는 약은?")와 제품명("판콜에이내복액") 사이의 의미적 유사도를 학습합니다.

---

## 🧩 2-Step Fine-tuning Pipeline

### 🔹 Step 1. Drug Type Semantic Alignment
| 항목 | 설명 |
|------|------|
| 데이터셋 | `data/drug_type_def_list.csv` |
| 목적 | `"해열제"` → `"체온을 낮추는 약"`과 같은 의약품 유형 의미 정렬 |
| 출력 | `model/fine_tuned_e5_small_drugtype` |
| 학습 스크립트 | `scripts/train_drug_type_e5_small.py` |

---

### 🔹 Step 2. Drug Product Semantic Alignment
| 항목 | 설명 |
|------|------|
| 데이터셋 | `data/drug_product_similarity_train.csv` |
| 목적 | 실제 제품(`판콜에이내복액`)과 자연어 질의(`열을 내리는 약`) 간 의미 매칭 학습 |
| 출력 | `model/fine_tuned_e5_small_drugproduct` |
| 학습 스크립트 | `scripts/train_drug_product_e5_small.py` |

---

## 💡 Example Usage

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

## ⚙️ Environment

| 항목 | 버전 |
|------|------|
| Python | 3.12.x |
| torch | 2.4.1 |
| transformers | 4.44.2 |
| sentence-transformers | 3.0.1 |
| accelerate | 0.27.0 |
| pandas | 2.2.3 |

설치 명령어:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
fine-tuned-e5-drugmodel/
├── data/
│   ├── drug_type_def_list.csv
│   ├── drug_type_similarity_train.csv
│   ├── drug_product_similarity_train.csv
│   └── drug_data_20_per_product_type.csv
│
├── scripts/
│   ├── make_drug_type_similarity_train.py
│   ├── train_drug_type_e5_small.py
│   ├── make_drug_product_similarity_train.py
│   └── train_drug_product_e5_small.py
│
├── model_cards/
│   └── README.md
│
├── utilities/
│   └── prepare_hf_readme_and_requirements.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Author & License

- Author: **@ryukato**
- Base model: `intfloat/multilingual-e5-small`
- License: [MIT License](https://opensource.org/licenses/MIT)
- Last updated: **2025-10-27**

---

## 🚀 Citation

If you use this model or training setup, please cite:

```bibtex
@software{ryukato_2025_fine_tuned_e5_small_drugmodel,
  author = {Yoon Y. Yoo},
  title = {Fine-tuned E5-small for Korean Drug Product Semantic Embedding},
  year = {2025},
  url = {https://huggingface.co/Yoonyoul/fine-tuned-e5-small-drugproduct}
}
```

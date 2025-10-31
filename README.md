# 🧬 Fine-tuned E5-small for Korean Drug Product Semantic Embedding

## 📘 Overview

이 저장소는 **[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)** 기반의  
한국어 의약품 의미 임베딩(Semantic Embedding) 모델을 구축하기 위한 전체 학습 파이프라인을 포함합니다.

이 프로젝트는 **의약품 유형(`drug_type_definition`) → DUR 규제/주의 정보(`dur_type_definition`) → 실제 제품(`drug_summary`, `drug_details`)** 으로 이어지는  
**3단계 파인튜닝(fine-tuning)** 과정을 통해,  
질의("열을 내리는 약은?", "임산부가 복용하면 안 되는 약은?")와 제품명("판콜에이내복액", "아목사펜캡슐") 사이의 의미적 유사도를 학습합니다.

[Model on HuggingFace](https://huggingface.co/Yoonyoul/fine-tuned-e5-small-drugproduct)

---

## 🧩 3-Step Fine-tuning Pipeline

### 🔹 Step 1. Drug Type Semantic Alignment
| 항목 | 설명 |
|------|------|
| 데이터셋 | `data/drug_type_similarity_train.csv` |
| 목적 | `"해열제"` → `"체온을 낮추는 약"`과 같은 **의약품 유형 의미 정렬** |
| 출력 | `model/fine_tuned_e5_small_drugtype` |
| 학습 스크립트 | `scripts/train_drug_type_e5_small.py` |

---

### 🔹 Step 2. DUR Type Semantic Alignment
| 항목 | 설명 |
|------|------|
| 데이터셋 | `data/drug_dur_type_similarity_train.csv` |
| 목적 | `"임부금기"`, `"노인주의"`, `"병용금기"` 등 **DUR 타입 용어와 전문적 설명** 간 의미 정렬 |
| 출력 | `model/fine_tuned_e5_small_drugdurtype` |
| 학습 스크립트 | `scripts/train_drug_durtype_e5_small.py` |

예시 데이터:
```csv
dur_type,description,label
임부금기,임산부에게 투여할 경우 태아에 부정적인 영향을 줄 수 있어 사용이 금지된 약물입니다.,1.0
병용금기,두 가지 이상의 약물을 함께 복용할 경우 심각한 부작용이나 상호작용이 발생할 수 있어 함께 사용이 금지된 약물입니다.,1.0
노인주의,노인의 생리적 특성과 대사 저하로 인해 부작용이 증가할 수 있어 투여 시 주의가 필요한 약물입니다.,1.0
```

이 단계에서는 **의약품 안전성과 DUR 규제 개념을 강화 학습**하여,  
“안전성 중심 질의(예: ‘임산부가 복용하면 안 되는 약’)”에도 강건한 의미 검색이 가능하도록 설계되었습니다.

---

### 🔹 Step 3. Drug Product Semantic Alignment
| 항목 | 설명 |
|------|------|
| 데이터셋 | `data/drug_product_similarity_train.csv` |
| 목적 | 실제 제품(`판콜에이내복액`)과 자연어 질의(`열을 내리는 약`) 간 의미 매칭 학습 |
| 출력 | `model/fine_tuned_e5_small_drugproduct_accum` |
| 학습 스크립트 | `scripts/train_drug_product_e5_small.py` |

---

## 🔹 Experimental: Drug Ingredient + Product Type Fine-tuning

`fine_tuned_e5_small_drugdurtype` 모델을 기반으로,  
의약품 성분(`ingredient_name`)과 제품 유형(`product_type`)을 결합한  
임베딩 학습(`fine_tuned_e5_small_drug_ptype_ingredients`)을 추가로 수행하였습니다.  

### ⚙️ 적용 내용
| 항목 | 값 |
|------|----|
| **학습 데이터** | `"성분명은(는) 제품유형 제제에서 사용되는 의약 성분이다."` |
| **샘플 사이즈** | 1,289 |
| **평균 손실** | 0.0012 |
| **유사도 평가** | 의미적 구분이 충분히 이루어지지 않음 |
| **관찰 예시** | “소염·진통제” 계열의 `이부프로펜`과 비관련 성분인 `염화나트륨`, `세티리진`이 모두 0.91~0.94 수준의 유사도를 보임 |

### 📋 관찰 내용
- 모델은 안정적으로 수렴하였으나, 문장 패턴의 반복성과 Positive-only 데이터 구성으로 인해  
  효능군 간 의미적 경계가 제대로 형성되지 않았음.  
- 전체 유사도 분포가 과도하게 높게 수렴하여, 의미보다 문체 패턴을 중심으로 학습된 것으로 관찰됨.  
- **결론적으로 본 모델은 성분-제품유형 결합 학습이 의미 검색 품질 개선에 실질적 이점을 제공하지 않음을 확인하였으며, 현재 파이프라인에는 적용하지 않기로 결정함.**

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
│   ├── drug_type_similarity_train.csv
│   ├── drug_dur_type_similarity_train.csv
│   ├── drug_product_similarity_train.csv
│
├── scripts/
│   ├── train_drug_type_e5_small.py
│   ├── train_drug_durtype_e5_small.py
│   ├── train_drug_product_e5_small.py
│   └── eval_model_rag_embedding.py
│
├── model_cards/
│   └── README.md
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

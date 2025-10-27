import os
import pandas as pd
import random
import numpy as np

# ✅ reproducibility (재현성 확보)
random.seed(42)
np.random.seed(42)

# ✅ 현재 스크립트 파일 기준으로 CSV 경로 계산
# 경로 설정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
csv_path = os.path.join(data_dir, "drug_type_def_list.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

# ✅ CSV 읽기
df = pd.read_csv(csv_path)

if "name" not in df.columns or "definition" not in df.columns:
    raise ValueError("CSV에는 'name'과 'definition' 컬럼이 포함되어야 합니다.")


# ✅ Paraphrase 질의 생성 함수
def generate_queries(type_name: str):
    """의약품 유형명(type_name)을 기반으로 자연어 질의 변형 생성"""
    return [
        type_name,
        f"{type_name}이란?",
        f"{type_name}은 어떤 약이야?",
        f"{type_name}의 효능은?",
        f"{type_name}은 언제 써?",
        f"{type_name} 관련 약 알려줘",
        f"{type_name}의 정의가 뭐야?",
    ]


rows = []

# ✅ Positive pairs (유사 pair)
for _, row in df.iterrows():
    type_name = str(row["name"]).strip()
    desc = str(row["definition"]).strip()

    for q in generate_queries(type_name):
        rows.append({"query": q, "context": desc, "label": 1.0})

# ✅ Negative pairs (비유사 pair)
for _, row in df.iterrows():
    type_name = str(row["name"]).strip()
    neg_desc_candidates = df[df["name"] != type_name]["definition"].tolist()
    if not neg_desc_candidates:
        continue
    neg_desc = random.choice(neg_desc_candidates)
    rows.append({"query": type_name, "context": neg_desc, "label": 0.0})

# ✅ 섞기
random.shuffle(rows)

# ✅ 저장
out_path = os.path.join(data_dir, "drug_type_similarity_train.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)

print(f"✅ 학습용 CSV 생성 완료 → {out_path}")
print(f"총 {len(rows)}개의 pair (positive + negative) 생성됨")

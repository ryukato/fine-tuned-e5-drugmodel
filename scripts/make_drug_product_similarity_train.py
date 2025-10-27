import os
import re
from typing import Optional
import pandas as pd
import random

# -----------------------------
# 경로 설정
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(base_dir, "resources")

input_path = os.path.join(resource_dir, "drug_data_20_per_product_type.csv")
output_path = os.path.join(resource_dir, "drug_product_similarity_train.csv")


# -----------------------------
# 헬퍼 함수
# -----------------------------
def extract_ingredient_names(raw: Optional[str]) -> str:
    """
    [M082272]디오스민 | [M012345]헤스페리딘 → '디오스민, 헤스페리딘'
    """
    if not str or not isinstance(raw, str) or not raw.strip():
        return ""
    # 대괄호 안의 코드 제거 후 파이프(|)를 쉼표로 변환
    cleaned = re.sub(r"\[[^\]]*\]", "", raw)
    cleaned = cleaned.replace("|", ",")
    # 불필요한 공백 정리
    return ", ".join([s.strip() for s in cleaned.split(",") if s.strip()])


# -----------------------------
# 데이터 로드 및 전처리
# -----------------------------
df = pd.read_csv(input_path)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 결측 제거
df = df.dropna(subset=["type_code", "type_name", "type_definition", "item_name"])
print(f"✅ 데이터 로드 완료: {len(df)} rows")

# -----------------------------
# 학습 데이터 구성
# -----------------------------
rows = []
for _, row in df.iterrows():
    type_code = row["type_code"]
    type_name = row["type_name"]
    type_def = row["type_definition"]
    item_name = row["item_name"]
    enterprise = row.get("enterprise_name", "")
    main_ingredient = extract_ingredient_names(row.get("main_ingredient", ""))
    chart = row.get("chart", "")
    # material = main_ingredient
    storage = row.get("storage_method", "")
    valid_term = row.get("valid_term", "")

    # context (제품 설명)
    context = " ".join(
        [
            f"{item_name}은(는) {enterprise}에서 제조한 {type_name} 계열의 의약품입니다.",
            f"{type_def}",
            f"주요 성분은 {main_ingredient} 입니다.",
            f"성상은 {chart}, 저장방법은 {storage}, 유효기간은 {valid_term}입니다.",
        ]
    ).strip()

    # Positive 쿼리들
    positive_queries = [
        f"{item_name}은 어떤 약이야?",
        f"{item_name}의 효능은?",
        f"{item_name}의 주요 성분은?",
        f"{item_name}은 언제 복용해?",
        f"{type_name} 약 알려줘",
    ]

    for q in positive_queries:
        rows.append({"query": q, "context": context, "label": 1.0})

    # Negative pair: 다른 type_code에서 무작위 선택
    neg_row = df[df["type_code"] != type_code].sample(1).iloc[0]
    neg_context = " ".join(
        [
            f"{neg_row['item_name']}은(는) {neg_row['enterprise_name']}에서 제조한 {neg_row['type_name']} 계열의 의약품입니다.",
            f"{neg_row['type_definition']}",
        ]
    )
    rows.append(
        {"query": f"{item_name}은 어떤 약이야?", "context": neg_context, "label": 0.0}
    )

# -----------------------------
# 셔플 및 저장
# -----------------------------
random.shuffle(rows)
pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ 제품 유사도 학습용 CSV 생성 완료 → {output_path}")
print(f"총 {len(rows)}개의 pair 생성 (positive + negative)")

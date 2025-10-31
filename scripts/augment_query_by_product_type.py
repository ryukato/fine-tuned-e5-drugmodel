import os
import pandas as pd
import random

# ----------------------------------------------------------------------
# 1. 효능 키워드 사전 (product_type → 관련 질의어)
# ----------------------------------------------------------------------
PRODUCT_TYPE_KEYWORDS = {
    "소염·진통제": ["소염 효과가 있는 성분", "통증 완화 약물", "염증 억제 성분", "진통 작용 물질"],
    "해열·진통제": ["열을 내리는 성분", "해열 효과가 있는 약물", "진통 효과가 있는 성분"],
    "항히스타민제": ["알레르기 완화 성분", "비염에 쓰이는 약물", "가려움 억제 약물"],
    "기타수액제": ["체액 보충 성분", "수분 공급 약물", "전해질 균형 유지 성분"],
    "강심제": ["심장 기능 강화 성분", "심부전 치료에 쓰이는 약물", "심근 수축을 돕는 성분"],
    "정장제": ["장 운동을 조절하는 성분", "소화 기능 개선 약물", "유산균 관련 제제"],
    "항생제": ["세균 감염 치료 성분", "항균 작용이 있는 약물", "감염 억제 약물"],
    "혈압강하제": ["혈압을 낮추는 성분", "고혈압 치료제의 주성분", "혈관을 확장하는 약물"],
    "진해거담제": ["기침을 완화하는 성분", "가래를 제거하는 약물", "호흡기 완화 성분"],
}

# ----------------------------------------------------------------------
# 2. 파일 경로 설정
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "../data")

input_csv = os.path.join(data_dir, "drug_product_type_ingredients_similarity_train.csv")
output_csv = os.path.join(data_dir, "drug_product_type_ingredients_augmented_train.csv")

# ----------------------------------------------------------------------
# 3. CSV 로드
# ----------------------------------------------------------------------
df = pd.read_csv(input_csv)
print(f"✅ Loaded {len(df)} base samples")

# ----------------------------------------------------------------------
# 4. 질의-문맥 쌍 생성
# ----------------------------------------------------------------------
rows = []
for _, row in df.iterrows():
    ptype = str(row["product_type"]).strip()
    ingredient = str(row["ingredient_name"]).strip()
    context = str(row["description"]).strip()

    keywords = PRODUCT_TYPE_KEYWORDS.get(ptype, [f"{ptype}의 주요 성분"])
    for q in random.sample(keywords, min(3, len(keywords))):  # 최대 3개 샘플링
        rows.append({
            "query": q,
            "context": context,
            "label": 1.0,
            "product_type": ptype,
            "ingredient_name": ingredient
        })

aug_df = pd.DataFrame(rows)
print(f"🚀 Generated {len(aug_df)} augmented samples")

# ----------------------------------------------------------------------
# 5. 저장
# ----------------------------------------------------------------------
aug_df.to_csv(output_csv, index=False)
print(f"✅ Saved → {output_csv}")
from sentence_transformers import SentenceTransformer, util

# 모델 로드
model = SentenceTransformer("model/fine_tuned_e5_small_drug_ptype_ingredients")

# 테스트 질의
query = "소염 효과가 있는 성분은?"

# 후보 문장들
docs = [
    "이부프로펜은 소염·진통제 제제에서 사용되는 의약 성분이다.",
    "염화나트륨은 기타수액제 제제에서 사용되는 의약 성분이다.",
    "세티리진은 항히스타민제 제제에서 사용되는 의약 성분이다."
]

# 임베딩
emb_q = model.encode(query, convert_to_tensor=True)
emb_docs = model.encode(docs, convert_to_tensor=True)

# 코사인 유사도 계산
scores = util.cos_sim(emb_q, emb_docs)[0]

# 한 줄씩 보기 좋게 출력
print(f"\n🔍 Query: {query}\n")
for doc, score in zip(docs, scores):
    print(f"{doc} → 유사도: {score.item():.4f}")
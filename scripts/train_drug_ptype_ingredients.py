import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from sentence_transformers import losses, SentenceTransformer, InputExample, util
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F

# ----------------------------------------------------------------------
# 1. 경로 설정
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(base_dir) == "scripts":
    base_dir = os.path.dirname(base_dir)

data_dir = os.path.join(base_dir, "data")
train_csv = os.path.join(data_dir, "drug_product_type_ingredients_augmented_train.csv")

base_model_path = os.path.join(base_dir, "model", "fine_tuned_e5_small_drugdurtype")
output_model_path = os.path.join(base_dir, "model", "fine_tuned_e5_small_drug_ptype_ingredients")
os.makedirs(output_model_path, exist_ok=True)
print(f"train_csv={train_csv}, output_model_path={output_model_path}")

# ----------------------------------------------------------------------
# 2. 환경 설정
# ----------------------------------------------------------------------
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cpu"
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print("⚙️ MPS available but using CPU for stability.")

# ----------------------------------------------------------------------
# 3. 모델 로드
# ----------------------------------------------------------------------
model = SentenceTransformer(base_model_path, device=device)
model.max_seq_length = 256

# ----------------------------------------------------------------------
# 4. 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(train_csv)
size = len(df)

# 현재 데이터에는 query/context/label 컬럼이 없으므로 생성
# ingredient_name을 query로, description을 context로 사용
if "query" not in df.columns:
    df["query"] = df["ingredient_name"]
if "context" not in df.columns:
    df["context"] = df["description"]
if "label" not in df.columns:
    df["label"] = 1.0  # 모든 쌍을 positive로 처리 (similarity 학습용)

# 데이터 샘플링
if size > 3000:
    df = df.sample(3000, random_state=42)
    size = len(df)
    print(f"📉 Sampled 3000 rows (original={len(df)}).")
else:
    print(f"⚠️ Data smaller than 3000 rows — using full dataset ({size} rows).")

# 에폭 결정
if size <= 1000:
    epochs = 1
elif size <= 3000:
    epochs = 1
elif size <= 10000:
    epochs = 2
else:
    epochs = 3
print(f"✅ Loaded {size} samples for training (epochs={epochs})")

# ----------------------------------------------------------------------
# 5. DataLoader 구성
# ----------------------------------------------------------------------
def collate_fn(batch):
    texts1 = [ex.texts[0] for ex in batch]
    texts2 = [ex.texts[1] for ex in batch]
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.float)
    features1 = model.tokenize(texts1)
    features2 = model.tokenize(texts2)
    return features1, features2, labels

train_examples = [
    InputExample(texts=[row.query, row.context], label=float(row.label))
    for row in df.itertuples()
]

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=2,
    num_workers=0,
    collate_fn=collate_fn,
)

# ----------------------------------------------------------------------
# 6. Loss, Optimizer 설정
# ----------------------------------------------------------------------
train_loss = losses.CosineSimilarityLoss(model)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
accum_steps = 4

# ----------------------------------------------------------------------
# 7. 학습 루프
# ----------------------------------------------------------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()
    print(f"\n🚀 Epoch {epoch+1}/{epochs} 시작")

    for step, (features1, features2, labels) in enumerate(tqdm(train_dataloader, leave=False)):
        features1 = {k: v.to(device) for k, v in features1.items()}
        features2 = {k: v.to(device) for k, v in features2.items()}
        labels = labels.to(device)

        emb1 = model.forward(features1)["sentence_embedding"]
        emb2 = model.forward(features2)["sentence_embedding"]

        emb1 = F.normalize(emb1)
        emb2 = F.normalize(emb2)

        cos_sim = torch.cosine_similarity(emb1, emb2)
        loss = F.mse_loss(cos_sim, labels)

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_dataloader)
    print(f"✅ Epoch {epoch+1} 완료 | 평균 손실: {avg_loss:.6f}")

# ----------------------------------------------------------------------
# 8. 모델 저장
# ----------------------------------------------------------------------
model.save(output_model_path)
print(f"\n✅ Fine-tuning 완료 → {output_model_path}")

# ----------------------------------------------------------------------
# 9. 간단 검증
# ----------------------------------------------------------------------
query = "통증 완화에 사용되는 성분은?"
emb_q = model.encode(query, convert_to_tensor=True)
emb_cands = model.encode(df["context"].head(10).tolist(), convert_to_tensor=True)
scores = util.cos_sim(emb_q, emb_cands)[0]

print("\n🔍 샘플 유사도 결과:")
for i, s in enumerate(scores):
    print(f"{df['ingredient_name'].iloc[i]} → {s.item():.4f}")
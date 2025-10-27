# file: tunning/train_drug_product_accum.py
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, util
import pandas as pd

# ----------------------------------------------------------------------
# 1. 경로 설정
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(base_dir, "resources")
train_csv = os.path.join(resource_dir, "drug_product_similarity_train.csv")

base_model_path = os.path.join(base_dir, "model", "fine_tuned_e5_small_drugtype")
output_model_path = os.path.join(
    base_dir, "model", "fine_tuned_e5_small_drugproduct_accum"
)
os.makedirs(output_model_path, exist_ok=True)

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
# 4. 데이터 로드 및 토크나이즈
# ----------------------------------------------------------------------
df = pd.read_csv(train_csv)
df = df.sample(3000, random_state=42)
print(f"✅ Loaded {len(df)} samples for training")


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
# 5. Loss, Optimizer
# ----------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3
accum_steps = 4

# ----------------------------------------------------------------------
# 6. 학습 루프
# ----------------------------------------------------------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()

    print(f"\n🚀 Epoch {epoch+1}/{epochs} 시작")
    for step, (features1, features2, labels) in enumerate(tqdm(train_dataloader)):
        features1 = {k: v.to(device) for k, v in features1.items()}
        features2 = {k: v.to(device) for k, v in features2.items()}
        labels = labels.to(device)

        # ✅ sentence embedding 추출
        emb1 = model.forward(features1)["sentence_embedding"]
        emb2 = model.forward(features2)["sentence_embedding"]

        cos_sim = torch.cosine_similarity(emb1, emb2)
        loss = torch.nn.functional.mse_loss(cos_sim, labels)

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(
        f"✅ Epoch {epoch+1} 완료 | 평균 손실: {total_loss / len(train_dataloader):.6f}"
    )

# ----------------------------------------------------------------------
# 7. 모델 저장
# ----------------------------------------------------------------------
model.save(output_model_path)
print(f"\n✅ Fine-tuning 완료 → {output_model_path}")

# ----------------------------------------------------------------------
# 8. 간단 검증
# ----------------------------------------------------------------------
query = "열을 내리는 약은?"
emb_q = model.encode(query, convert_to_tensor=True)
emb_cands = model.encode(df["context"].head(10).tolist(), convert_to_tensor=True)
scores = util.cos_sim(emb_q, emb_cands)[0]

print("\n🔍 샘플 유사도 결과:")
for i, s in enumerate(scores):
    print(f"{df['query'].iloc[i][:40]} → {s.item():.4f}")

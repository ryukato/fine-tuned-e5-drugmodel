import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, util

# ----------------------------------------------------------------------
# 1. 경로 설정
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))  # 루트에서 실행 시에도 안전
if os.path.basename(base_dir) == "scripts":
    base_dir = os.path.dirname(base_dir)

data_dir = os.path.join(base_dir, "data")
train_csv = os.path.join(data_dir, "drug_type_similarity_train.csv")

output_model_path = os.path.join(base_dir, "model", "fine_tuned_e5_small_drugtype")
os.makedirs(output_model_path, exist_ok=True)
print(f"📂 train_csv={train_csv}\n📦 output_model_path={output_model_path}")

# ----------------------------------------------------------------------
# 2. 환경 설정
# ----------------------------------------------------------------------
"""
> Note
> This configuration is for Apple Silicon (MPS). 
> If you use CUDA (GPU), comment out the following lines and set device='cuda'.
"""
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
base_model_path = "intfloat/multilingual-e5-small"  # Hugging Face pretrained model
model = SentenceTransformer(base_model_path, device=device)
model.max_seq_length = 256

# ----------------------------------------------------------------------
# 4. 데이터 로드 및 전처리
# ----------------------------------------------------------------------
df = pd.read_csv(train_csv)
size = len(df)

# 안전한 샘플링
if size > 3000:
    df = df.sample(3000, random_state=42)
    size = len(df)
else:
    print(f"⚠️ Data smaller than 3000 rows — using full dataset ({size} rows).")

# ✅ 데이터 크기별 epoch 설정 (명시적, 캐시 방지)
if size <= 1000:
    epochs = 1
elif size <= 3000:
    epochs = 1
elif size <= 10000:
    epochs = 2
else:
    epochs = 3

print(f"✅ Loaded {size} samples for training (epochs={epochs})")

# InputExample 구성
train_examples = [
    InputExample(texts=[row.query, row.context], label=float(row.label))
    for row in df.itertuples()
]
print(f"✅ Prepared {len(train_examples)} training pairs")

# Collate function
def collate_fn(batch):
    texts1 = [ex.texts[0] for ex in batch]
    texts2 = [ex.texts[1] for ex in batch]
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.float)
    features1 = model.tokenize(texts1)
    features2 = model.tokenize(texts2)
    return features1, features2, labels

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=2,  # 안전한 메모리 설정
    num_workers=0,
    collate_fn=collate_fn,
)

# ----------------------------------------------------------------------
# 5. Optimizer
# ----------------------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
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

        emb1 = model.forward(features1)["sentence_embedding"]
        emb2 = model.forward(features2)["sentence_embedding"]

        cos_sim = torch.cosine_similarity(emb1, emb2)
        loss = torch.nn.functional.mse_loss(cos_sim, labels)

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"✅ Epoch {epoch+1} 완료 | 평균 손실: {total_loss / len(train_dataloader):.6f}")

# ----------------------------------------------------------------------
# 7. 모델 저장
# ----------------------------------------------------------------------
model.save(output_model_path)
print(f"\n✅ Fine-tuning 완료 → {output_model_path}")

# ----------------------------------------------------------------------
# 8. 간단 검증
# ----------------------------------------------------------------------
query = "해열제는 어떤 약이야?"
emb_q = model.encode(query, convert_to_tensor=True)
emb_cands = model.encode(df["context"].head(10).tolist(), convert_to_tensor=True)
scores = util.cos_sim(emb_q, emb_cands)[0]

print("\n🔍 샘플 유사도 결과:")
for i, s in enumerate(scores):
    print(f"{df['query'].iloc[i][:40]} → {s.item():.4f}")
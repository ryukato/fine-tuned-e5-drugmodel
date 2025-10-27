import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# 1️⃣ 경로 설정
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(base_dir, "resources")
train_csv = os.path.join(resource_dir, "drug_product_similarity_train.csv")

base_model_path = os.path.join(base_dir, "model", "fine_tuned_e5_small_drugtype")
output_dir = os.path.join(base_dir, "model", "fine_tuned_e5_small_product")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 2️⃣ 환경 설정
# ----------------------------------------------------------------------
if torch.backends.mps.is_available():
    print("⚙️ MPS available but using CPU for stability.")
device = "cpu"

# ----------------------------------------------------------------------
# 3️⃣ 모델 로드 + encoder freeze
# ----------------------------------------------------------------------
model = SentenceTransformer(base_model_path, device=device)

# encoder freeze (optional but reduces memory)
for param in model._first_module().parameters():
    param.requires_grad = False

# ----------------------------------------------------------------------
# 4️⃣ 데이터 로드
# ----------------------------------------------------------------------
df = pd.read_csv(train_csv)
train_examples = [
    InputExample(texts=[row.query, row.context], label=float(row.label))
    for row in df.itertuples()
]

print(f"✅ Loaded {len(train_examples)} pairs")

# ----------------------------------------------------------------------
# 5️⃣ DataLoader & Loss
# ----------------------------------------------------------------------
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=2,  # ultra safe
    num_workers=0,  # prevent process forking
)
train_loss = losses.CosineSimilarityLoss(model)

# ----------------------------------------------------------------------
# 6️⃣ 학습
# ----------------------------------------------------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=int(0.1 * len(train_dataloader)),
    show_progress_bar=True,
    output_path=output_dir,
)

print(f"✅ Fine-tuning 완료 — 모델 저장됨 → {output_dir}")

# ----------------------------------------------------------------------
# 7️⃣ 간단 검증
# ----------------------------------------------------------------------
query = "해열제는 어떤 약이야?"
contexts = df["context"].sample(5).tolist()
emb_q = model.encode(query, convert_to_tensor=True)
emb_c = model.encode(contexts, convert_to_tensor=True)
scores = util.cos_sim(emb_q, emb_c)[0]

for c, s in zip(contexts, scores):
    print(f"\n{c[:80]}... → {s.item():.4f}")

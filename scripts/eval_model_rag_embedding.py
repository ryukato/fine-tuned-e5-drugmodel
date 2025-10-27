import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------------------------
# 1️⃣ 인자 설정
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate fine-tuned RAG embedding models.")
parser.add_argument("--model", type=str, required=True, help="Model directory path")
parser.add_argument("--data", type=str, required=True, help="CSV data path for evaluation")
parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
args = parser.parse_args()

model_path = os.path.abspath(args.model)
data_path = os.path.abspath(args.data)

print(f"📂 model={model_path}")
print(f"📘 eval_data={data_path}")
print(f"⚙️ Using device: {args.device}")

# ----------------------------------------------------------------------
# 2️⃣ 모델 & 데이터 로드
# ----------------------------------------------------------------------
model = SentenceTransformer(model_path, device=args.device)
df = pd.read_csv(data_path)

# query/context 컬럼 확인
expected_cols = {"query", "context", "label"}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"❌ CSV must include columns: {expected_cols}")

print(f"✅ Loaded {len(df)} evaluation pairs")

# ----------------------------------------------------------------------
# 3️⃣ 임베딩 생성
# ----------------------------------------------------------------------
print("🔍 Encoding embeddings...")
query_embs = model.encode(df["query"].tolist(), convert_to_tensor=True, show_progress_bar=False)
context_embs = model.encode(df["context"].tolist(), convert_to_tensor=True, show_progress_bar=False)

# cosine similarity (diagonal = true pair)
sims = util.cos_sim(query_embs, context_embs)
true_scores = sims.diag().cpu().numpy()

mean_sim = np.mean(true_scores)
std_sim = np.std(true_scores)
print(f"✅ Mean Cosine Similarity: {mean_sim:.4f} ± {std_sim:.4f}")

# ----------------------------------------------------------------------
# 4️⃣ Retrieval 평가 (Top-k / MRR)
# ----------------------------------------------------------------------
def retrieval_metrics(sims, k=5):
    top_k_acc, reciprocal_ranks = 0, []
    sims_np = sims.cpu().numpy()
    for i in tqdm(range(len(sims_np))):
        ranks = np.argsort(-sims_np[i])  # descending
        if i in ranks[:k]:
            top_k_acc += 1
        rank_position = np.where(ranks == i)[0]
        reciprocal_ranks.append(1 / (rank_position[0] + 1) if rank_position.size > 0 else 0)
    return (top_k_acc / len(sims_np)) * 100, np.mean(reciprocal_ranks)

top5_acc, mrr = retrieval_metrics(sims, k=5)

# ----------------------------------------------------------------------
# 5️⃣ 결과 출력
# ----------------------------------------------------------------------
print("\n📊 Evaluation Summary")
print(f"Top-5 Accuracy : {top5_acc:.2f}%")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Average Cosine Similarity : {mean_sim:.4f}")

# ----------------------------------------------------------------------
# 6️⃣ 샘플 결과 확인
# ----------------------------------------------------------------------
print("\n🔎 Sample Predictions:")
for i in range(5):
    q, c, s = df['query'][i], df['context'][i], true_scores[i]
    print(f"[{i+1}] {q[:40]} ↔ {c[:70]} → {s:.4f}")
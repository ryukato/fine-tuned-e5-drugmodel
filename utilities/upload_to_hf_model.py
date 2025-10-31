import os
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, login
from sentence_transformers import util

# ----------------------------------------------------------------------
# 0️⃣ 인자 설정
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Upload model or README to Hugging Face Hub")
parser.add_argument(
    "--mode",
    choices=["model", "readme"],
    default="model",
    help="Select upload mode: 'model' for model push, 'readme' for README update only",
)
args = parser.parse_args()

# ----------------------------------------------------------------------
# 1️⃣ 환경 로드 및 로그인
# ----------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or input("🔑 Enter your Hugging Face Token: ").strip()
login(token=HF_TOKEN)

repo_id = "Yoonyoul/fine-tuned-e5-small-drugproduct"
local_model_path = "model/fine_tuned_e5_small_drugproduct_accum"
readme_path = "model_cards/README.md"

api = HfApi()

# ----------------------------------------------------------------------
# 2️⃣ 업로드 모드에 따른 분기
# ----------------------------------------------------------------------

if args.mode == "model":
    print(f"📦 Loading model from: {local_model_path}")
    model = SentenceTransformer(local_model_path)

    print(f"🚀 Pushing model to Hugging Face Hub → {repo_id}")
    model.push_to_hub(
        repo_id,
        commit_message="Upload fine-tuned e5-small model (drugproduct)",
        exist_ok=True,  # 기존 모델 덮어쓰기 허용
    )

    print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_id}")

    # README는 모델 업로드 이후에 덮어쓰기
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="🧾 Update model card with project-specific README (after model push)",
    )

elif args.mode == "readme":
    print("🧾 Updating only README.md (no model push)")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="🧾 Update model card with project-specific README only",
    )
    print(f"✅ README successfully updated for {repo_id}")

else:
    raise ValueError("Invalid mode. Use --mode model or --mode readme.")

# ----------------------------------------------------------------------
# 3️⃣ (선택) 테스트 로드 (모델 모드일 때만)
# ----------------------------------------------------------------------
if args.mode == "model":
    print("\n🧠 Testing load from hub...")
    model_test = SentenceTransformer(repo_id)
    query = "열을 내리는 약은?"
    docs = [
        "판콜에이내복액은 해열진통제입니다.",
        "마이암부톨정은 항결핵제입니다.",
        "루센티스주사는 습성 황반변성 치료제입니다."
    ]

    emb_q = model_test.encode(query, convert_to_tensor=True)
    emb_d = model_test.encode(docs, convert_to_tensor=True)
    scores = util.cos_sim(emb_q, emb_d)[0]

    print("\n🔍 Sample Similarity Results:")
    for d, s in zip(docs, scores):
        print(f"{d} → {s.item():.4f}")
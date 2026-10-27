"""Deploy to a Hugging Face Space (Docker SDK): create it, set secrets from .env, upload the code.

Log in once first: `hf auth login` (token with write access). Then:
    python -m scripts.deploy_space --space GRVA07/investigation-engine
upload_folder is used instead of `git push` because Spaces reject plain-git binary files (PDF, DOCX).
Secret values are read from .env and sent only to the Space; they are never printed.
"""
import argparse

from huggingface_hub import HfApi

from app.config import get_settings

SECRETS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "QDRANT_URL", "QDRANT_API_KEY", "COLLECTION")
IGNORE = [".env", ".git/*", ".venv/*", "**/__pycache__/*", ".pytest_cache/*", "data/qdrant/*",
          "data/qdrant_subset/*", "data/raw/*", "data/*.jsonl", "data/*.log"]


def secret_values() -> dict[str, str]:
    s = get_settings()
    values = {"LLM_API_KEY": s.llm_api_key, "LLM_BASE_URL": s.llm_base_url, "LLM_MODEL": s.llm_model,
              "QDRANT_URL": s.qdrant_url, "QDRANT_API_KEY": s.qdrant_api_key, "COLLECTION": s.collection}
    missing = [k for k in SECRETS if not values[k]]
    if missing:
        raise SystemExit(f"Fill these in .env first: {', '.join(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="user/space-name")
    args = parser.parse_args()
    values = secret_values()
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key, value in values.items():
        api.add_space_secret(args.space, key, value)
    print(f"Secrets set: {', '.join(values)}")
    api.upload_folder(folder_path=".", repo_id=args.space, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy from local repository")
    print(f"Uploaded. Build log: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()

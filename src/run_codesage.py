import os
import json
import time
import subprocess
from pathlib import Path
import logging
import argparse
import numpy as np
import torch
import transformers
import sentence_transformers
import sys

from sentence_transformers import SentenceTransformer

from run_bm25 import load_config, ensure_dir, repo_dir_name, load_index, save_output, clone_repo, compute_ground_truth
from index_builder import build_repo_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str)
    parser.add_argument("--commit", type=str)
    parser.add_argument("--model", type=str, default=os.environ.get("CODESAGE_MODEL", r"/home/jiaxin/codesage-small"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dev = str(args.device).lower()
    if dev == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    elif dev.startswith("cuda"):
        idx = dev.split(":", 1)[1] if ":" in dev else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        device = "cuda"
    else:
        device = dev
    if device == "cuda":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        try:
            logging.info(f"torch_cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} current_device={(torch.cuda.current_device() if torch.cuda.is_available() else -1)} device_name={(torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count()>0 else '')} visible={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
        except Exception:
            logging.warning("cuda diagnostics failed")
    try:
        logging.info(f"lib_versions transformers={getattr(transformers,'__version__',None)} sentence_transformers={getattr(sentence_transformers,'__version__',None)} torch={getattr(torch,'__version__',None)} py={sys.version.split()[0]}")
    except Exception:
        pass
    try:
        from transformers.modeling_utils import Conv1D as _Conv1D
    except Exception as e_conv:
        try:
            from transformers.models.gpt2.modeling_gpt2 import Conv1D as GPT2Conv1D
            import transformers.modeling_utils as mu
            setattr(mu, 'Conv1D', GPT2Conv1D)
            logging.info("patched Conv1D from GPT2 into modeling_utils")
        except Exception as e2:
            logging.warning(f"patch Conv1D failed: {e2}")
    cfg = load_config()
    dataset_dir = Path(cfg["dataset_dir"])
    repos_dir = Path(cfg["repos_dir"])
    indexes_dir = Path(cfg["indexes_dir"])
    outputs_dir = Path(cfg["outputs_dir"]) / "codesage"
    bins_dir = Path(cfg.get("bins_dir", "bin"))
    logs_dir = Path(cfg["logs_dir"]) / "codesage"
    ensure_dir(repos_dir)
    ensure_dir(indexes_dir)
    ensure_dir(outputs_dir)
    ensure_dir(bins_dir)
    ensure_dir(logs_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(str(logs_dir / "run.log"), encoding="utf-8"), logging.StreamHandler()], force=True)
    files = [dataset_dir / p for p in cfg["dataset_files"]]
    include_extensions = cfg["indexing"]["include_extensions"]
    exclude_dirs = cfg["indexing"]["exclude_dirs"]
    max_file_size_bytes = cfg["indexing"]["max_file_size_bytes"]
    top_k = cfg["retrieval"]["top_k"]
    base_url = cfg["clone"]["base_url"]
    do_clone = cfg["clone"]["enable"]
    all_items = []
    for p in files:
        logging.info(f"loading dataset {p}")
        import pandas as pd
        df = pd.read_parquet(p)
        id_col = "instance_id" if "instance_id" in df.columns else ("id" if "id" in df.columns else None)
        repo_col = "repo" if "repo" in df.columns else ("repository" if "repository" in df.columns else ("repo_name" if "repo_name" in df.columns else None))
        commit_col = "commit" if "commit" in df.columns else ("base_commit" if "base_commit" in df.columns else ("repo_commit" if "repo_commit" in df.columns else ("hash" if "hash" in df.columns else None)))
        title_col = "issue_title" if "issue_title" in df.columns else ("title" if "title" in df.columns else None)
        body_col = "problem_statement" if "problem_statement" in df.columns else ("issue_body" if "issue_body" in df.columns else ("body" if "body" in df.columns else ("description" if "description" in df.columns else ("issue_description" if "issue_description" in df.columns else None))))
        patch_col = "patch" if "patch" in df.columns else ("gold_patch" if "gold_patch" in df.columns else ("diff" if "diff" in df.columns else None))
        split_name = "dev" if "dev" in str(p).lower() else "test"
        for _, row in df.iterrows():
            q = ""
            if title_col and body_col:
                q = f"{row[title_col] or ''}\n{row[body_col] or ''}"
            elif body_col:
                q = row[body_col] or ""
            all_items.append({"instance_id": row[id_col] if id_col else None, "repo": row[repo_col] if repo_col else None, "commit": row[commit_col] if commit_col else None, "query": q, "patch": row[patch_col] if patch_col else None, "split": split_name})
    if args.repo:
        all_items = [it for it in all_items if it["repo"] == args.repo]
    if args.commit:
        all_items = [it for it in all_items if it["commit"] == args.commit]
    groups = {}
    for it in all_items:
        key = (it["repo"], it["commit"]) if it["repo"] and it["commit"] else None
        if key:
            groups.setdefault(key, []).append(it)
    encoder = None
    model_load_error = None
    mp = Path(args.model)
    logging.info(f"loading model from {args.model} exists={mp.exists()}")
    try:
        encoder = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    except Exception as e:
        model_load_error = e
        logging.exception("model load failed for provided path")
        encoder = None
    for (repo, commit), items in groups.items():
        dname = repo_dir_name(repo, commit)
        rdir = repos_dir / dname
        if do_clone:
            ok = True
            if not rdir.exists():
                base_prefix = repo.replace("/", "__") + "_"
                try:
                    candidates = [d for d in repos_dir.iterdir() if d.is_dir() and d.name.startswith(base_prefix)]
                except Exception:
                    candidates = []
                if candidates:
                    import shutil, random, stat
                    try:
                        src_dir = random.choice(candidates)
                        logging.info(f"reuse repo {src_dir} -> {rdir}")
                        shutil.copytree(str(src_dir), str(rdir))
                        for root, dirs, files in os.walk(str(rdir)):
                            for name in dirs:
                                p = os.path.join(root, name)
                                try:
                                    os.chmod(p, stat.S_IWRITE)
                                except Exception:
                                    pass
                            for name in files:
                                p = os.path.join(root, name)
                                try:
                                    os.chmod(p, stat.S_IWRITE)
                                except Exception:
                                    pass
                        env = os.environ.copy()
                        env["GIT_TERMINAL_PROMPT"] = "0"
                        try:
                            subprocess.run(["git", "-C", str(rdir), "checkout", "--detach", "--force", commit], check=True, env=env, input=("y\n" * 200), text=True)
                        except Exception:
                            subprocess.run(["git", "-C", str(rdir), "reset", "--hard", commit], check=True, env=env, input=("y\n" * 200), text=True)
                    except Exception:
                        try:
                            shutil.rmtree(str(rdir), ignore_errors=True)
                        except Exception:
                            pass
                        logging.info(f"cloning repo {repo} at {commit} -> {rdir}")
                        ok = clone_repo(base_url, repo, commit, str(rdir))
                else:
                    logging.info(f"cloning repo {repo} at {commit} -> {rdir}")
                    ok = clone_repo(base_url, repo, commit, str(rdir))
            else:
                logging.info(f"cloning repo {repo} at {commit} -> {rdir}")
                ok = clone_repo(base_url, repo, commit, str(rdir))
                if ok:
                    try:
                        logging.info(f"checkout {commit} in {rdir}")
                        env = os.environ.copy()
                        env["GIT_TERMINAL_PROMPT"] = "0"
                        subprocess.run(["git", "-C", str(rdir), "checkout", "--detach", "--force", commit], check=True, env=env, input=("y\n" * 200), text=True)
                    except Exception:
                        ok = False
            if not ok:
                logging.warning(f"clone failed for {repo}@{commit}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "clone_failed"})
                continue
        else:
            if not rdir.exists():
                logging.info(f"skip missing local repo {rdir}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "repo_missing"})
                continue
            try:
                logging.info(f"checkout {commit} in {rdir}")
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"
                subprocess.run(["git", "-C", str(rdir), "checkout", "--detach", "--force", commit], check=True, env=env, input=("y\n" * 200), text=True)
            except Exception:
                logging.warning(f"checkout failed for {repo}@{commit}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "checkout_failed"})
                continue
        try:
            env = os.environ.copy()
            env["GIT_TERMINAL_PROMPT"] = "0"
            head = subprocess.check_output(["git", "-C", str(rdir), "rev-parse", "HEAD"], env=env).decode().strip()
        except Exception:
            head = None
        if (not head) or (commit and head.lower() != commit.lower()):
            logging.warning(f"head mismatch for {repo}@{commit} in {rdir}: {head}")
            for it in items:
                save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "checkout_mismatch"})
            continue
        logging.info(f"checkout到{commit}开始检索")
        index_file = indexes_dir / f"{dname}.jsonl"
        need_build = (not index_file.exists()) or (index_file.exists() and os.path.getsize(index_file) == 0)
        if need_build:
            try:
                logging.info(f"building index for {rdir}")
                _, total = build_repo_index(str(rdir), str(indexes_dir), max_file_size_bytes, include_extensions, exclude_dirs)
                logging.info(f"indexed {total} functions for {dname}")
            except Exception:
                logging.warning(f"index failed for {dname}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "index_failed"})
                continue
        docs = load_index(index_file)
        total = len(docs)
        if total == 0:
            try:
                logging.info(f"rebuilding empty index for {dname}")
                _, total = build_repo_index(str(rdir), str(indexes_dir), max_file_size_bytes, include_extensions, exclude_dirs)
                docs = load_index(index_file)
                total = len(docs)
            except Exception:
                logging.warning(f"index rebuild failed for {dname}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "index_failed"})
                continue
        if encoder is None:
            logging.warning("model load failed")
            err = {"error": "model_load_failed", "exception": (str(model_load_error) if model_load_error else None), "device": device, "model": args.model, "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE")}
            try:
                err["model_files_sample"] = sorted([p.name for p in mp.iterdir()])[:20]
            except Exception:
                pass
            for it in items:
                save_output(bins_dir / it["split"], it, [], 0, 0.0, err)
            continue
        texts = [d["text"] for d in docs]
        try:
            logging.info(f"encoding docs total={len(texts)} device={device} batch_size={args.batch_size} repo={dname}")
            t0 = time.time()
            doc_embs = encoder.encode(texts, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
            logging.info(f"encoded {len(texts)} docs in {time.time() - t0:.3f}s for {dname}")
        except Exception as e:
            logging.exception(f"encode failed for {dname}")
            for it in items:
                save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "encode_failed", "exception": str(e)})
            continue
        for it in items:
            gt_files, gt_locs = compute_ground_truth(it.get("patch"), str(rdir))
            it["ground_truth_files"] = gt_files
            it["ground_truth_locations"] = gt_locs
            logging.info(f"gt files={len(gt_files)} locs={len(gt_locs)} for {it['instance_id']}")
            q = it["query"] or ""
            try:
                q_emb = encoder.encode([q], batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)[0]
            except Exception as e:
                logging.exception(f"query encode failed for {it['instance_id']}")
                for it2 in [it]:
                    save_output(bins_dir / it2["split"], it2, [], total, 0.0, {"error": "query_encode_failed", "exception": str(e)})
                continue
            t0 = time.time()
            scores = np.dot(doc_embs, q_emb)
            idx = np.argsort(-scores)[:top_k]
            dt = time.time() - t0
            preds = [docs[i]["doc_id"] for i in idx]
            raw = [{"doc_id": docs[i]["doc_id"], "score": float(scores[i])} for i in idx]
            target_dir = (outputs_dir / it["split"]) if gt_locs else (bins_dir / it["split"]) 
            save_output(target_dir, it, preds, total, dt, raw)


if __name__ == "__main__":
    main()

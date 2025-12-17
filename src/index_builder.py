import os
import json
import re
import ast
import argparse
import logging
import subprocess
import time
from pathlib import Path
import pandas as pd


def tokenize(text):
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def normalize_relpath(p):
    return Path(p).as_posix()


def iter_files(root, exclude_dirs, include_extensions):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if any(fn.endswith(ext) for ext in include_extensions):
                yield Path(dirpath) / fn


def extract_segment(lines, node):
    s = max(getattr(node, "lineno", 1) - 1, 0)
    e = getattr(node, "end_lineno", None)
    if e is None:
        e = s + 1
        while e < len(lines) and lines[e].strip() != "":
            e += 1
    return "".join(lines[s:e])


def build_repo_index(repo_dir, indexes_dir, max_file_size_bytes, include_extensions, exclude_dirs):
    repo_dir = Path(repo_dir)
    index_path = Path(indexes_dir) / f"{repo_dir.name}.jsonl"
    meta_path = Path(indexes_dir) / f"{repo_dir.name}.meta.json"
    docs = []
    total = 0
    os.makedirs(indexes_dir, exist_ok=True)
    for fpath in iter_files(repo_dir, set(exclude_dirs), include_extensions):
        try:
            if os.path.getsize(fpath) > max_file_size_bytes:
                continue
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
            lines = src.splitlines(keepends=True)
            tree = ast.parse(src)
        except Exception:
            continue
        rel = normalize_relpath(fpath.relative_to(repo_dir))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code = extract_segment(lines, node)
                prefix = f"文件路径: {rel}\n"
                text = prefix + code
                tokens = tokenize(text)
                doc_id = f"{rel}::{node.name}"
                docs.append({
                    "doc_id": doc_id,
                    "file_path": rel,
                    "class_name": None,
                    "func_name": node.name,
                    "text": text,
                    "tokens": tokens,
                    "doc_len": len(tokens),
                })
                total += 1
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        code = extract_segment(lines, sub)
                        prefix = f"文件路径: {rel}, 类名: {node.name}\n"
                        text = prefix + code
                        tokens = tokenize(text)
                        doc_id = f"{rel}::{node.name}.{sub.name}"
                        docs.append({
                            "doc_id": doc_id,
                            "file_path": rel,
                            "class_name": node.name,
                            "func_name": sub.name,
                            "text": text,
                            "tokens": tokens,
                            "doc_len": len(tokens),
                        })
                        total += 1
    with open(index_path, "w", encoding="utf-8") as out:
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8") as m:
        json.dump({"total_functions_indexed": total}, m, ensure_ascii=False)
    return index_path, total


def load_config():
    with open(Path(".config"), "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def select_field(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clone_repo(base_url, repo, commit, dest):
    if Path(dest).exists():
        return True
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        subprocess.run(["git", "clone", "--no-checkout", f"{base_url}/{repo}.git", dest], check=True, env=env)
        subprocess.run(["git", "-C", dest, "checkout", "--detach", "--force", commit], check=True, env=env)
        return True
    except Exception:
        try:
            subprocess.run(["git", "-C", dest, "reset", "--hard", commit], check=True, env=env)
            return True
        except Exception:
            return False


def load_instances(parquet_path):
    df = pd.read_parquet(parquet_path)
    repo_col = select_field(df, ["repo", "repository", "repo_name"])
    commit_col = select_field(df, ["commit", "base_commit", "repo_commit", "hash"])
    items = []
    for _, row in df.iterrows():
        repo = row[repo_col] if repo_col else None
        commit = row[commit_col] if commit_col else None
        items.append({"repo": repo, "commit": commit})
    return items


def repo_dir_name(repo, commit):
    base = repo.replace("/", "__")
    return f"{base}_{commit}" if commit else base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--repo", type=str)
    parser.add_argument("--commit", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    cfg = load_config()
    dataset_dir = Path(cfg["dataset_dir"])
    repos_dir = Path(cfg["repos_dir"])
    indexes_dir = Path(cfg["indexes_dir"])
    logs_dir = Path(cfg["logs_dir"]) / "index"
    ensure_dir(repos_dir)
    ensure_dir(indexes_dir)
    ensure_dir(logs_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(str(logs_dir / "run.log"), encoding="utf-8"), logging.StreamHandler()])
    files = [dataset_dir / p for p in cfg["dataset_files"]]
    include_extensions = cfg["indexing"]["include_extensions"]
    exclude_dirs = cfg["indexing"]["exclude_dirs"]
    max_file_size_bytes = cfg["indexing"]["max_file_size_bytes"]
    base_url = cfg["clone"]["base_url"]
    do_clone = cfg["clone"]["enable"]
    if args.path:
        dname = Path(args.path).name
        rdir = Path(args.path)
        logging.info(f"building index for explicit path {rdir}")
        try:
            _, total = build_repo_index(str(rdir), str(indexes_dir), max_file_size_bytes, include_extensions, exclude_dirs)
            logging.info(f"indexed {total} functions for {dname}")
        except Exception:
            logging.warning(f"index failed for {dname}")
        logging.info("done 1/1")
        return
    items = []
    for p in files:
        logging.info(f"loading dataset {p}")
        items.extend(load_instances(p))
    keys = {}
    for it in items:
        if it["repo"] and it["commit"]:
            if args.repo and it["repo"] != args.repo:
                continue
            if args.commit and it["commit"] != args.commit:
                continue
            keys[(it["repo"], it["commit"])] = True
    total_repos = len(keys)
    logging.info(f"repos to process: {total_repos}")
    processed = 0
    for repo, commit in keys.keys():
        dname = repo_dir_name(repo, commit)
        rdir = repos_dir / dname
        index_file = indexes_dir / f"{dname}.jsonl"
        if do_clone:
            logging.info(f"cloning {repo}@{commit} -> {rdir}")
            ok = clone_repo(base_url, repo, commit, str(rdir))
            if not ok:
                logging.warning(f"clone failed for {repo}@{commit}")
                processed += 1
                continue
        else:
            if not rdir.exists():
                logging.info(f"skip missing local repo {rdir}")
                processed += 1
                continue
            try:
                logging.info(f"checkout {commit} in {rdir}")
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"
                subprocess.run(["git", "-C", str(rdir), "checkout", "--detach", "--force", commit], check=True, env=env)
            except Exception:
                logging.warning(f"checkout failed for {repo}@{commit}")
                processed += 1
                continue
        if index_file.exists() and not args.rebuild:
            logging.info(f"skip existing index {index_file}")
            processed += 1
            continue
        try:
            logging.info(f"building index for {dname}")
            _, total = build_repo_index(str(rdir), str(indexes_dir), max_file_size_bytes, include_extensions, exclude_dirs)
            logging.info(f"indexed {total} functions for {dname}")
        except Exception:
            logging.warning(f"index failed for {dname}")
        processed += 1
    logging.info(f"done {processed}/{total_repos}")


if __name__ == "__main__":
    main()

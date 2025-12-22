import os
import json
import time
import subprocess
from pathlib import Path
import logging
import argparse
import re
import shutil
import random
import stat

import pandas as pd

from bm25 import BM25
from index_builder import build_repo_index, tokenize


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
    Path(dest).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "-C", dest, "init"], check=True, env=env)
        subprocess.run(["git", "-C", dest, "remote", "add", "origin", f"git@github.com:{repo}.git"], check=True, env=env)
        subprocess.run(["git", "-C", dest, "fetch", "--depth", "1", "--no-tags", "origin", commit], check=True, env=env)
        subprocess.run(["git", "-C", dest, "checkout", "--detach", "--force", "FETCH_HEAD"], check=True, env=env, input=("y\n" * 200), text=True)
        return True
    except Exception:
        try:
            subprocess.run(["git", "-C", dest, "fetch", "--no-tags", "origin", commit], check=True, env=env)
            subprocess.run(["git", "-C", dest, "checkout", "--detach", "--force", "FETCH_HEAD"], check=True, env=env, input=("y\n" * 200), text=True)
            return True
        except Exception:
            try:
                subprocess.run(["git", "clone", "--no-checkout", f"git@github.com:{repo}.git", dest], check=True, env=env)
                subprocess.run(["git", "-C", dest, "fetch", "--depth", "1", "--no-tags", "origin", commit], check=True, env=env)
                subprocess.run(["git", "-C", dest, "checkout", "--detach", "--force", "FETCH_HEAD"], check=True, env=env, input=("y\n" * 200), text=True)
                return True
            except Exception:
                return False


def load_instances(parquet_path):
    df = pd.read_parquet(parquet_path)
    id_col = select_field(df, ["instance_id", "id"])
    repo_col = select_field(df, ["repo", "repository", "repo_name"])
    commit_col = select_field(df, ["commit", "base_commit", "repo_commit", "hash"])
    title_col = select_field(df, ["issue_title", "title"])
    body_col = select_field(df, ["problem_statement", "issue_body", "body", "description", "issue_description"])
    patch_col = select_field(df, ["patch", "gold_patch", "diff"])
    instances = []
    for _, row in df.iterrows():
        iid = row[id_col] if id_col else None
        repo = row[repo_col] if repo_col else None
        commit = row[commit_col] if commit_col else None
        q = ""
        if title_col and body_col:
            q = f"{row[title_col] or ''}\n{row[body_col] or ''}"
        elif body_col:
            q = row[body_col] or ""
        patch = row[patch_col] if patch_col else None
        instances.append({"instance_id": iid, "repo": repo, "commit": commit, "query": q, "patch": patch})
    return instances


def repo_dir_name(repo, commit):
    base = repo.replace("/", "__")
    return f"{base}_{commit}" if commit else base


def load_index(index_path):
    docs = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def save_output(out_dir, item, preds, total, dt, raw):
    ensure_dir(out_dir)
    out = {
        "instance_id": item["instance_id"],
        "query": item["query"],
        "ground_truth": {"files": item.get("ground_truth_files", []), "locations": item.get("ground_truth_locations", [])},
        "predictions": preds,
        "metadata": {
            "repo": item["repo"],
            "commit": item["commit"],
            "total_functions_indexed": total,
            "retrieval_time_seconds": dt,
        },
        "raw_output": json.dumps(raw, ensure_ascii=False),
    }
    p = Path(out_dir) / f"{item['instance_id']}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str)
    parser.add_argument("--commit", type=str)
    args = parser.parse_args()
    cfg = load_config()
    dataset_dir = Path(cfg["dataset_dir"])
    repos_dir = Path(cfg["repos_dir"])
    indexes_dir = Path(cfg["indexes_dir"])
    outputs_dir = Path(cfg["outputs_dir"]) / "bm25"
    bins_dir = Path(cfg.get("bins_dir", "bin"))
    logs_dir = Path(cfg["logs_dir"]) / "bm25"
    ensure_dir(repos_dir)
    ensure_dir(indexes_dir)
    ensure_dir(outputs_dir)
    ensure_dir(bins_dir)
    ensure_dir(logs_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(str(logs_dir / "run.log"), encoding="utf-8"), logging.StreamHandler()])
    files = [dataset_dir / p for p in cfg["dataset_files"]]
    include_extensions = cfg["indexing"]["include_extensions"]
    exclude_dirs = cfg["indexing"]["exclude_dirs"]
    max_file_size_bytes = cfg["indexing"]["max_file_size_bytes"]
    top_k = cfg["retrieval"]["top_k"]
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    base_url = cfg["clone"]["base_url"]
    do_clone = cfg["clone"]["enable"]
    all_items = []
    for p in files:
        logging.info(f"loading dataset {p}")
        items = load_instances(p)
        split_name = "dev" if "dev" in str(p).lower() else "test"
        for it in items:
            it["split"] = split_name
        all_items.extend(items)
    if args.repo:
        all_items = [it for it in all_items if it["repo"] == args.repo]
    if args.commit:
        all_items = [it for it in all_items if it["commit"] == args.commit]
    groups = {}
    for it in all_items:
        key = (it["repo"], it["commit"]) if it["repo"] and it["commit"] else None
        if key:
            groups.setdefault(key, []).append(it)
    for (repo, commit), items in groups.items():
        dname = repo_dir_name(repo, commit)
        rdir = repos_dir / dname
        rdir_existed_before = rdir.exists()
        if do_clone:
            ok = True
            if not rdir.exists():
                base_prefix = repo.replace("/", "__") + "_"
                candidates = []
                if candidates:
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
        index_file = indexes_dir / f"{dname}.jsonl"
        meta_file = indexes_dir / f"{dname}.meta.json"
        force_rebuild = True
        if force_rebuild:
            try:
                logging.info(f"rebuilding index for {dname}")
                _, total = build_repo_index(str(rdir), str(indexes_dir), max_file_size_bytes, include_extensions, exclude_dirs)
                logging.info(f"indexed {total} functions for {dname}")
            except Exception:
                logging.warning(f"index failed for {dname}")
                for it in items:
                    save_output(bins_dir / it["split"], it, [], 0, 0.0, {"error": "index_failed"})
                continue
        else:
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
        model = BM25(docs, k1=k1, b=b)
        for it in items:
            q_tokens = tokenize(it["query"] or "")
            gt_files, gt_locs = compute_ground_truth(it.get("patch"), str(rdir))
            it["ground_truth_files"] = gt_files
            it["ground_truth_locations"] = gt_locs
            logging.info(f"gt files={len(gt_files)} locs={len(gt_locs)} for {it['instance_id']}")
            t0 = time.time()
            scored = model.topk(q_tokens, k=top_k)
            dt = time.time() - t0
            preds = [d["doc_id"] for s, d in scored]
            raw = [{"doc_id": d["doc_id"], "score": float(s)} for s, d in scored]
            target_dir = (outputs_dir / it["split"]) if gt_locs else (bins_dir / it["split"]) 
            save_output(target_dir, it, preds, total, dt, raw)


# entrypoint moved to bottom after helper definitions


# Helpers for ground truth extraction

def _strip_ab_prefix(path):
    if not path:
        return None
    path = path.strip()
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse_patch_files_and_hunks(patch_text):
    files = set()
    hunks = []
    if not patch_text:
        return list(files), hunks
    current_file_base = None
    current_file_new = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            current_file_base = None
            current_file_new = None
        elif line.startswith("--- "):
            p = line[4:].strip()
            if p != "/dev/null":
                current_file_base = _strip_ab_prefix(p)
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                current_file_new = _strip_ab_prefix(p)
            # prefer base path when available
            f = current_file_base or current_file_new
            if f:
                files.add(f)
        elif line.startswith("@@"):
            m = re.match(r"@@\s*-([0-9]+)(?:,([0-9]+))?\s*\+([0-9]+)(?:,([0-9]+))?\s*@@", line)
            if m:
                minus_start = int(m.group(1))
                minus_len = int(m.group(2) or 1)
                # use base path to map ranges onto base code
                f = current_file_base or current_file_new
                if f:
                    hunks.append({"file": f, "minus_start": minus_start, "minus_len": minus_len})
    return list(files), hunks


def compute_ground_truth(patch_text, repo_dir):
    files, hunks = parse_patch_files_and_hunks(patch_text)
    gt_files = []
    gt_locs = []
    for f in files:
        fp = Path(repo_dir) / f
        if fp.exists():
            gt_files.append(f)
    # map hunks to functions
    cache_spans = {}
    for h in hunks:
        f = h["file"]
        fp = Path(repo_dir) / f
        if not fp.exists():
            continue
        if f not in cache_spans:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
                spans = _function_spans(src)
                cache_spans[f] = spans
            except Exception:
                cache_spans[f] = []
        start = h["minus_start"]
        end = start + max(h["minus_len"], 1) - 1
        for sp in cache_spans[f]:
            if _ranges_intersect((start, end), (sp["start"], sp["end"])):
                if sp["class_name"] and sp["func_name"]:
                    doc_id = f"{f}::{sp['class_name']}.{sp['func_name']}"
                elif sp["class_name"] and not sp["func_name"]:
                    doc_id = f"{f}::{sp['class_name']}"
                elif sp["func_name"]:
                    doc_id = f"{f}::{sp['func_name']}"
                else:
                    doc_id = f
                gt_locs.append(doc_id)
    # dedupe
    gt_locs = list(dict.fromkeys(gt_locs))
    return gt_files, gt_locs


def _function_spans(src):
    import ast
    spans = []
    try:
        tree = ast.parse(src)
        lines = src.splitlines()
    except Exception:
        return spans
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append({"class_name": None, "func_name": node.name, "start": getattr(node, "lineno", 1), "end": getattr(node, "end_lineno", getattr(node, "lineno", 1))})
        elif isinstance(node, ast.ClassDef):
            spans.append({"class_name": node.name, "func_name": None, "start": getattr(node, "lineno", 1), "end": getattr(node, "end_lineno", getattr(node, "lineno", 1))})
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append({"class_name": node.name, "func_name": sub.name, "start": getattr(sub, "lineno", 1), "end": getattr(sub, "end_lineno", getattr(sub, "lineno", 1))})
    return spans


def _ranges_intersect(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


if __name__ == "__main__":
    main()

import argparse
import json
import os
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd


def _write_marti_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def export_gsm_hard_marti_from_parquet(train_parquet: str, test_parquet: str, out_train_jsonl: str, out_test_jsonl: str) -> None:
    def _convert_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for idx, row in df.reset_index(drop=True).iterrows():
            prompt = "You need to write python program to solve math problems:\n" + str(row.get("input", ""))
            answer = str(row.get("target", ""))
            converted.append({"prompt": prompt, "answer": answer, "metadata": {"idx": int(idx)}})
        return converted

    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    _write_marti_jsonl(out_train_jsonl, _convert_df(train_df))
    _write_marti_jsonl(out_test_jsonl, _convert_df(test_df))


def export_srdd_marti_from_csv(train_csv: str, test_csv: str, out_train_jsonl: str, out_test_jsonl: str) -> None:
    def _convert_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for idx, row in df.reset_index(drop=True).iterrows():
            prompt = "Develop a pythonic software following description:\n" + str(row.get("Description", ""))
            converted.append({"prompt": prompt, "answer": "", "metadata": {"idx": int(idx), "name": str(row.get("Name", "")), "category": str(row.get("Category", ""))}})
        return converted

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    _write_marti_jsonl(out_train_jsonl, _convert_df(train_df))
    _write_marti_jsonl(out_test_jsonl, _convert_df(test_df))


def export_scibench_marti_from_jsonl(train_jsonl: str, test_jsonl: str, out_train_jsonl: str, out_test_jsonl: str) -> None:
    instruction = (
        "Solve the following science problem.\n"
        "Return only the final numeric answer (do not include units unless explicitly asked).\n\n"
    )

    def _convert_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            prompt = instruction + str(item.get("problem_text", ""))
            answer = str(item.get("answer_number", ""))
            meta = {
                "idx": idx,
                "unit": item.get("unit", ""),
                "source": item.get("source", ""),
                "problemid": item.get("problemid", str(idx)),
            }
            converted.append({"prompt": prompt, "answer": answer, "metadata": meta})
        return converted

    train_items = _read_jsonl(train_jsonl)
    test_items = _read_jsonl(test_jsonl)
    _write_marti_jsonl(out_train_jsonl, _convert_items(train_items))
    _write_marti_jsonl(out_test_jsonl, _convert_items(test_items))


def _split_indices(n: int, train_size: int, seed: int) -> Tuple[List[int], List[int]]:
    if n <= 0:
        return [], []
    train_size = min(train_size, n)
    perm = pd.Series(range(n)).sample(frac=1, random_state=seed).tolist()
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    return train_idx, test_idx


def split_parquet(input_path: str, out_train: str, out_test: str, train_size: int, seed: int) -> Tuple[int, int]:
    df = pd.read_parquet(input_path)
    train_idx, test_idx = _split_indices(len(df), train_size, seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)
    return len(train_df), len(test_df)


def split_csv(input_path: str, out_train: str, out_test: str, train_size: int, seed: int) -> Tuple[int, int]:
    df = pd.read_csv(input_path)
    train_idx, test_idx = _split_indices(len(df), train_size, seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    return len(train_df), len(test_df)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def split_jsonl(input_path: str, out_train: str, out_test: str, train_size: int, seed: int) -> Tuple[int, int]:
    items = _read_jsonl(input_path)
    train_idx, test_idx = _split_indices(len(items), train_size, seed)
    train_items = [items[i] for i in train_idx]
    test_items = [items[i] for i in test_idx]
    _write_jsonl(out_train, train_items)
    _write_jsonl(out_test, test_items)
    return len(train_items), len(test_items)


def _maybe_backup(path: str, backup_suffix: str, overwrite: bool) -> Optional[str]:
    if not os.path.exists(path):
        return None
    if overwrite:
        return None
    backup_path = path + backup_suffix
    if os.path.exists(backup_path):
        return backup_path
    os.rename(path, backup_path)
    return backup_path


def split_all(data_root: str, train_size: int, seed: int, overwrite: bool) -> None:
    gsm_dir = os.path.join(data_root, "GSM-Hard")
    srdd_dir = os.path.join(data_root, "SRDD")
    scibench_dir = os.path.join(data_root, "scibench")

    gsm_in = os.path.join(gsm_dir, "test.parquet")
    gsm_in_bak = os.path.join(gsm_dir, "test.parquet.bak")
    if os.path.exists(gsm_in) or os.path.exists(gsm_in_bak):
        _maybe_backup(os.path.join(gsm_dir, "train.parquet"), ".bak", overwrite)
        _maybe_backup(os.path.join(gsm_dir, "test.parquet"), ".bak", overwrite)
        split_parquet(
            input_path=gsm_in if (overwrite and os.path.exists(gsm_in)) else (gsm_in if os.path.exists(gsm_in) else gsm_in_bak),
            out_train=os.path.join(gsm_dir, "train.parquet"),
            out_test=os.path.join(gsm_dir, "test.parquet"),
            train_size=train_size,
            seed=seed,
        )

        # Export MARTI-compatible jsonl (prompt/answer)
        train_parquet = os.path.join(gsm_dir, "train.parquet")
        test_parquet = os.path.join(gsm_dir, "test.parquet")
        if os.path.exists(train_parquet) and os.path.exists(test_parquet):
            export_gsm_hard_marti_from_parquet(
                train_parquet=train_parquet,
                test_parquet=test_parquet,
                out_train_jsonl=os.path.join(gsm_dir, "train.jsonl"),
                out_test_jsonl=os.path.join(gsm_dir, "test.jsonl"),
            )

    srdd_in = os.path.join(srdd_dir, "SRDD.csv")
    if os.path.exists(srdd_in):
        _maybe_backup(os.path.join(srdd_dir, "train.csv"), ".bak", overwrite)
        _maybe_backup(os.path.join(srdd_dir, "test.csv"), ".bak", overwrite)
        split_csv(
            input_path=srdd_in,
            out_train=os.path.join(srdd_dir, "train.csv"),
            out_test=os.path.join(srdd_dir, "test.csv"),
            train_size=train_size,
            seed=seed,
        )

        # Export MARTI-compatible jsonl (prompt/answer)
        train_csv = os.path.join(srdd_dir, "train.csv")
        test_csv = os.path.join(srdd_dir, "test.csv")
        if os.path.exists(train_csv) and os.path.exists(test_csv):
            export_srdd_marti_from_csv(
                train_csv=train_csv,
                test_csv=test_csv,
                out_train_jsonl=os.path.join(srdd_dir, "train.jsonl"),
                out_test_jsonl=os.path.join(srdd_dir, "test.jsonl"),
            )

    scibench_in = os.path.join(scibench_dir, "train.jsonl")
    scibench_in_bak = os.path.join(scibench_dir, "train.jsonl.bak")
    if os.path.exists(scibench_in) or os.path.exists(scibench_in_bak):
        _maybe_backup(os.path.join(scibench_dir, "train.jsonl"), ".bak", overwrite)
        _maybe_backup(os.path.join(scibench_dir, "test.jsonl"), ".bak", overwrite)
        split_jsonl(
            input_path=scibench_in if (overwrite and os.path.exists(scibench_in)) else (scibench_in if os.path.exists(scibench_in) else scibench_in_bak),
            out_train=os.path.join(scibench_dir, "train.jsonl"),
            out_test=os.path.join(scibench_dir, "test.jsonl"),
            train_size=train_size,
            seed=seed,
        )

        # Export MARTI-compatible jsonl (prompt/answer)
        train_jsonl = os.path.join(scibench_dir, "train.jsonl")
        test_jsonl = os.path.join(scibench_dir, "test.jsonl")
        if os.path.exists(train_jsonl) and os.path.exists(test_jsonl):
            export_scibench_marti_from_jsonl(
                train_jsonl=train_jsonl,
                test_jsonl=test_jsonl,
                out_train_jsonl=os.path.join(scibench_dir, "train_marti.jsonl"),
                out_test_jsonl=os.path.join(scibench_dir, "test_marti.jsonl"),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    split_all(data_root=data_root, train_size=args.train_size, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

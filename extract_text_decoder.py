"""
google/translategemma-4b-it （マルチモーダル）から
テキストデコーダ部分だけを抽出して保存するスクリプト。

safetensors をメモリマップで開くので、RAMをほとんど使わない。
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

REPO_ID = "google/translategemma-4b-it"
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "./translategemma-text-only"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. config.json をダウンロードしてテキスト用に変換 ---
    print("=== config.json ダウンロード ===")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        full_config = json.load(f)

    # テキストデコーダ用のconfigを取り出す
    text_config = full_config.get("text_config", {})
    # トップレベルの必要な値も追加
    text_config["model_type"] = "gemma3_text"
    text_config["architectures"] = ["Gemma3ForCausalLM"]
    text_config["bos_token_id"] = full_config.get("bos_token_id", text_config.get("bos_token_id", 2))
    text_config["eos_token_id"] = full_config.get("eos_token_id", text_config.get("eos_token_id", 1))
    text_config["pad_token_id"] = full_config.get("pad_token_id", text_config.get("pad_token_id", 0))

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(text_config, f, indent=2)
    print(f"  config.json 保存完了 (model_type={text_config['model_type']})")

    # --- 2. tokenizer と generation_config をコピー ---
    print("\n=== tokenizer ダウンロード ===")
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=fname)
            import shutil
            shutil.copy2(path, os.path.join(OUTPUT_DIR, fname))
            print(f"  {fname} コピー完了")
        except Exception as e:
            print(f"  {fname} スキップ: {e}")

    # --- 3. safetensors インデックスをダウンロード ---
    print("\n=== safetensors インデックス ダウンロード ===")
    index_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # テキストデコーダの重みだけフィルタ
    # マルチモーダル: model.language_model.model.layers.* → テキスト: model.layers.*
    text_weights = {}
    for key, shard_file in weight_map.items():
        if key.startswith("model.language_model."):
            # model.language_model.model.xxx → model.xxx
            # model.language_model.lm_head.xxx → lm_head.xxx
            new_key = key.replace("model.language_model.", "")
            text_weights[new_key] = shard_file

    total = len(weight_map)
    text_count = len(text_weights)
    skip_count = total - text_count
    print(f"  全パラメータ: {total}")
    print(f"  テキストデコーダ: {text_count} (抽出対象)")
    print(f"  ビジョン等: {skip_count} (スキップ)")

    # --- 4. 必要なshardファイルだけダウンロード ---
    needed_shards = set(text_weights.values())
    print(f"\n=== safetensors ダウンロード ({len(needed_shards)} ファイル) ===")

    shard_paths = {}
    for shard in sorted(needed_shards):
        print(f"  {shard} ダウンロード中...")
        shard_paths[shard] = hf_hub_download(repo_id=REPO_ID, filename=shard)
        print(f"  完了")

    # --- 5. テキストデコーダの重みだけ抽出して保存 ---
    print("\n=== テキストデコーダ抽出中 ===")
    extracted = {}
    for new_key, shard_file in text_weights.items():
        old_key = "model.language_model." + new_key
        shard_path = shard_paths[shard_file]

        # メモリマップで開く（RAMを使わない）
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if old_key in f.keys():
                extracted[new_key] = f.get_tensor(old_key)

    print(f"  抽出完了: {len(extracted)} テンソル")

    # 保存（1ファイルにまとめる）
    output_safetensors = os.path.join(OUTPUT_DIR, "model.safetensors")
    print(f"\n=== model.safetensors 保存中 ===")
    save_file(extracted, output_safetensors)

    size_gb = os.path.getsize(output_safetensors) / 1024 / 1024 / 1024
    print(f"  保存完了: {size_gb:.2f} GB")

    # safetensors index も作り直す
    new_weight_map = {k: "model.safetensors" for k in extracted.keys()}
    new_index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in extracted.values())},
        "weight_map": new_weight_map,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # --- 6. 最終確認 ---
    print("\n=== 抽出結果 ===")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size / 1024 / 1024:.1f} MB")

    print("\n✅ テキストデコーダ抽出完了！")


if __name__ == "__main__":
    main()

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
    # まずキー名のパターンを確認
    all_keys = list(weight_map.keys())
    print(f"\n  最初の10キー（デバッグ）:")
    for k in all_keys[:10]:
        print(f"    {k}")

    # language_model のプレフィックスを自動検出
    # 可能なパターン: "model.language_model.", "language_model.", その他
    prefix = None
    for candidate in ["model.language_model.", "language_model."]:
        count = sum(1 for k in all_keys if k.startswith(candidate))
        if count > 0:
            prefix = candidate
            print(f"\n  検出されたプレフィックス: '{prefix}' ({count} 件)")
            break

    if prefix is None:
        # language_model が見つからない場合、model.layers があるか確認（既にテキストのみかも）
        has_layers = sum(1 for k in all_keys if k.startswith("model.layers."))
        if has_layers > 0:
            print(f"\n  ⚠️ 既にテキストデコーダ形式です（model.layers.* が {has_layers} 件）")
            prefix = ""  # プレフィックス無し = そのままコピー
        else:
            print(f"\n  ❌ テキストデコーダのキーが見つかりません")
            print(f"  全ユニークプレフィックス:")
            prefixes = set(k.split(".")[0] for k in all_keys)
            for p in sorted(prefixes):
                print(f"    {p}.* : {sum(1 for k in all_keys if k.startswith(p + '.'))}")
            sys.exit(1)

    text_weights = {}
    for key, shard_file in weight_map.items():
        if prefix == "":
            # 既にテキスト形式 → 全てコピー
            text_weights[key] = shard_file
        elif key.startswith(prefix):
            new_key = key[len(prefix):]  # プレフィックスを除去
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




    # --- 6. 最終確認 ---
    print("\n=== 抽出結果 ===")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size / 1024 / 1024:.1f} MB")

    print("\n✅ テキストデコーダ抽出完了！")


if __name__ == "__main__":
    main()

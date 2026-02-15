"""
onnxruntime-genai の model builder (gemma.py) にパッチを当てる。

TranslateGemma の config.json は rope_local_base_freq ではなく
rope_parameters.sliding_attention.rope_theta に値が入っているため、
builder がクラッシュする問題を修正する。
"""

import importlib
import onnxruntime_genai.models.builders.gemma as gemma_module

gemma_path = gemma_module.__file__
print(f"パッチ対象: {gemma_path}")

with open(gemma_path, "r") as f:
    source = f.read()

original = source

# 1. rope_local_base_freq → rope_parameters から取得
source = source.replace(
    "self.rope_local_theta = config.rope_local_base_freq",
    "self.rope_local_theta = getattr(config, 'rope_local_base_freq', None) or config.rope_parameters.get('sliding_attention', {}).get('rope_theta', 10000)",
)

# 2. rope_global_base_freq → rope_parameters から取得
source = source.replace(
    "self.rope_global_theta = config.rope_global_base_freq",
    "self.rope_global_theta = getattr(config, 'rope_global_base_freq', None) or config.rope_parameters.get('full_attention', {}).get('rope_theta', 1000000)",
)

# 3. sliding_window_pattern → _sliding_window_pattern フォールバック (issue #1826)
source = source.replace(
    "config.sliding_window_pattern",
    "getattr(config, 'sliding_window_pattern', None) or getattr(config, '_sliding_window_pattern', 6)",
)

if source != original:
    with open(gemma_path, "w") as f:
        f.write(source)
    print("✅ パッチ適用完了")
else:
    print("⚠️ パッチ対象が見つかりませんでした")

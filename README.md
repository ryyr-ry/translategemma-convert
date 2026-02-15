# TranslateGemma ONNX 変換

Google公式 `translategemma-4b-it` を `onnxruntime-genai` 用 INT4 ONNX に変換する。

## 使い方

1. Settings → Secrets → `HF_TOKEN` に Hugging Face の read トークンを登録
2. Actions タブ → 「TranslateGemma → ONNX INT4 変換」→ Run workflow
3. 完了したら Artifacts から `translategemma-genai-int4` をダウンロード

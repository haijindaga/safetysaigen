# Ubuntu 側セットアップ手順 (clone してすぐ動く)

対象: Ubuntu 22.04 / suzutaro 機 (RTX 4060 8GB, Ollama 稼働中)

## 1. clone と venv

```bash
git clone https://github.com/<user>/<repo>.git core-safety
cd core-safety
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q          # 24 passed になること
```

## 2. 2D シミュレーション動作確認 (GPU 不要)

```bash
python scripts/run_sim2d.py --scenario cone_barrier --method core --plot
python scripts/eval_baselines.py --repeats 5
```

## 3. VLM 評価 (Ollama)

Ollama が http://localhost:11434 で gemma3:27b を提供していること。

評価用画像を用意する (最初は 2D シミュレータから生成できる):

```bash
python scripts/make_vlm_dataset.py          # assets/vlm_eval/ に合成画像+ラベル生成
python scripts/eval_vlm_ollama.py --data assets/vlm_eval --model gemma3:27b --smoke
python scripts/eval_vlm_ollama.py --data assets/vlm_eval --model gemma3:27b
```

**注意**: 合成画像は疑似カラーの単純形状なので、実 VLM がクラスを認識できる
とは限らない。この段階の目的は **接続確認・JSON 出力の頑健性・レイテンシ計測**。
論文 Table II 相当の意味理解評価は、Isaac Sim のスクリーンショットか実写真を
`assets/vlm_eval/images/` に置き、`labels.json` に期待述語を書いて行うこと
(リポジトリの合成データはそのままフォーマット見本になる)。

- `--smoke` は 3 枚だけ流す接続確認。
- gemma3:27b は 8GB VRAM に載り切らず CPU に溢れるため、1 枚あたり
  数十秒かかる想定 (論文は L40 で 4.1s)。短い出力なので実用範囲。
- 比較したければ `--model gemma3:12b` も可 (論文 Table II の Small 行に対応)。

## 4. SAM3 (GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-ubuntu.txt
python scripts/test_sam3.py     # 合成画像で SAM3Segmenter の自己診断
```

- 初回実行時に facebook/sam3 の重み (~3.4GB) を HuggingFace から自動DL。
- fp16 で動かす設定になっている。VRAM が足りない場合は
  `SAM3Segmenter(device="cpu")` に切り替え (遅いが動く)。
- VLM は Ollama プロセス側の VRAM/CPU を使うため、SAM3 (fp16 ~4GB) と
  同時稼働できる見込み。両方張り付く場合は Ollama 側を CPU 寄りにする
  (`OLLAMA_NUM_GPU=0` など) か SAM3 を CPU に。

## 5. 既知の注意点

- `transformers>=5.0` が SAM3 対応版。古いと `Sam3Model` が無い。
- Ollama の応答が JSON 以外を含んでも `parse_vlm_output` がフェンス除去・
  末尾カンマ許容でパースする。完全に壊れた出力は「推論失敗」として
  カウントされる (論文の Llava 行と同じ扱い)。

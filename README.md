# core-safety

文脈依存のロボット安全フレームワークの実装。
事前の地図・安全仕様なしで、文脈依存の安全性を **推論 (VLM) → 接地 (セグメンテーション+空間オペレータ+コストマップ) → 強制 (CBF-QP)** の3段で実現する。
CBF は平面速度指令 `(vx, vy, ω)` に作用するため、移動ロボット・四足・モバイルマニピュレータで共通に使える。

## 構成

```
core_safety/
  predicates.py        述語 ON/NEAR/AROUND/BETWEEN と VLM JSON パーサ
  pipeline.py          3モジュールを束ねるオーケストレータ
  reasoning/
    prompt.py          VLM システムプロンプト (参照手法の Listing 1 準拠)
    vlm_client.py      OllamaVLM (gemma3:27b) / RuleBasedVLM (開発・ベースライン用)
  grounding/
    segmentation.py    Segmenter IF + 真値セグメンタ (2D sim / Isaac GT 用)
    sam3_segmenter.py  SAM3 (facebook/sam3, transformers) — Ubuntu GPU 用
    operators.py       空間オペレータ (AROUND=50px膨張, BETWEEN=凸包)
    image_safe_set.py  画像空間安全集合 (Eq. 2)
    projection.py      ピンホール逆投影 (Eq. 16, 3–7 m クリップ)
    costmap.py         0.2 m グリッド, P[safe]=n_s/(n_s+n_u), τ=0.5 (Eq. 3)
    barrier.py         符号付き距離場バリア h(x) + 解析勾配 (Eq. 4, 8)
  control/
    dynamics.py        平面制御アフィンモデル (Eq. 12)
    cbf_qp.py          CBF-QP (Eq. 7, CVXPY, α(x)=0.25x)
    nominal.py         ウェイポイント追従の名目制御器
  theory/
    certificate.py     Theorem 1 の安全走行証明書 (κ*_MTS 最適化, 付録B-D)
  sim2d/               レイキャストRGB-D+真値ラベルの2D閉ループシミュレータ
scripts/
  run_sim2d.py         1エピソード実行+プロット
  eval_baselines.py    12シナリオ × {core, nocontext, geometric} × Nシード (Table I 相当)
  eval_vlm_ollama.py   Ollama で VLM 安全推論を評価 (Table II/III 相当, Ubuntu)
tests/                 24 ユニットテスト (演算子/コストマップ/バリア/CBF/証明書/カメラ)
```

## セットアップ (Windows / Ubuntu 共通)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Ubuntu: source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

Ubuntu で VLM / SAM3 を使う場合は追加で:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-ubuntu.txt   # transformers (SAM3), accelerate
```

## 使い方

```bash
# シナリオ一覧
python scripts/run_sim2d.py --list

# 1エピソード (プロットは results/ に保存)
python scripts/run_sim2d.py --scenario cone_barrier --method core --plot
python scripts/run_sim2d.py --scenario grass_shortcut --method geometric --plot

# 評価マトリクス (Table I 相当)
python scripts/eval_baselines.py --repeats 5

# VLM 評価 (Ubuntu, Ollama 稼働中に)
python scripts/eval_vlm_ollama.py --data assets/vlm_eval --model gemma3:27b --smoke
```

## 参照手法との対応

| 参照手法 | 本実装 |
|---|---|
| VLM: 4bit Gemma 3 27B on Ollama | `OllamaVLM("gemma3:27b")` (同一構成) |
| プロンプト (Listing 1) | `reasoning/prompt.py` (逐語) |
| セグメンテーション: SAM3 | `grounding/sam3_segmenter.py` (facebook/sam3) |
| AROUND 膨張カーネル 50px | `operators.py` |
| コストマップ 0.2 m, τ=0.5 | `costmap.py` |
| 深度クリップ 3–7 m | `projection.py` |
| α(x)=0.25x, CVXPY | `cbf_qp.py` |
| 力学 (Eq. 12), Δt=0.1s, v≤0.35 | `dynamics.py` |
| Theorem 1 証明書 (κ*=3, δ=0.1) | `theory/certificate.py` (κ*=3 を再現) |
| 12タスク (safe 6 / unsafe 6) | `sim2d/scenarios.py` (2D 版) |
| ベースライン Oracle/NoContext/Geometric | `sim2d/runner.py` |

実装上の差異 (理由付き):

- **SDF のゼロ点をセル境界に置く半セルオフセット** — セル中心基準だと境界で
  ±1セルのチャタリングが起き、1セル分 (0.2 m) 侵入するため。
- **未観測セルは safe 扱い** — Assumption 1 (知覚が進入前に unsafe を検出する)
  と整合。全 unsafe 事前分布ではロボットが初期状態から動けない。
- **離散時間バックトラックチェック** — h(x) がセル単位の区分定数のため、
  連続時間 CBF 条件だけでは薄い unsafe 領域を複数ステップかけて
  すり抜けられる (実験で確認)。予測次状態で h(x_{t+1}) ≥ 0 を検証し、
  満たすまで入力を縮小する (u=0 で必ず充足)。論文が引く離散時間
  CBF 実装の議論 (ref. [54] Brunke et al. 2024) に沿った対処。
- **Windows 開発時は RuleBasedVLM + 真値セグメンタ** — VLM/セグメンテーションを
  決定的な代替に差し替えて接地・制御を単体検証する。Ubuntu では同一 IF で
  OllamaVLM + SAM3Segmenter に差し替わる。

## ロードマップ

- [x] Phase 0: コアライブラリ + ユニットテスト (Windows)
- [x] Phase 1: 2D 閉ループ再現実験 (Windows)
- [x] Phase 2: Ollama VLM 評価スクリプト + SAM3 アダプタ (Ubuntu で実行)
- [ ] Phase 3: Isaac Sim 統合 (env_isaaclab): 移動ロボット → 四足 → モバイルマニピュレータ

# Phase 3: Isaac Sim 統合 (env_isaaclab)

第一弾を実装済み: `scripts/isaac_mobile_demo.py` (Jetbot + コーン列)。
実行方法:

```bash
conda activate env_isaaclab
cd core-safety
pip install -r requirements.txt          # 初回のみ (torch には触れない)
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/isaac_mobile_demo.py                  # GT セグ + ルールブック推論
python scripts/isaac_mobile_demo.py --vlm ollama     # 実 VLM (非同期スレッド)
python scripts/isaac_mobile_demo.py --segmenter sam3 # 実 SAM3
python scripts/isaac_mobile_demo.py --headless       # GUI なし
```

期待挙動: 名目制御はゴール (x=5) へ直進しようとするが、コーン列 (x=3) の
手前でフィルタが停止させる (ログの `filtered=True`, `h` が 0 付近で停止)。

**注意 (未検証コード)**: このスクリプトは Windows 側では実行確認できない。
Isaac Sim はバージョンごとに API 名が変わるため (4.5 で omni.isaac.* →
isaacsim.* に改称)、両系統の import フォールバックを入れてあるが、
手元のバージョンで import エラーが出たら、そのモジュール名だけ読み替えて
ほしい (エラーメッセージのモジュール名を `docs.isaacsim.omniverse.nvidia.com`
の該当バージョンの API リファレンスで引くのが早い)。
Jetbot の USD パスも 5.x (`/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd`) と
4.x (`/Isaac/Robots/Jetbot/jetbot.usd`) で異なる — スクリプト冒頭の候補を
入れ替えれば対応できる。

CORE 側のインターフェースは 2D シミュレータと完全に同じなので、
Isaac 側は「RGB-D + ポーズを渡し、速度指令を受け取る」ブリッジだけである
(`core_safety/isaac/adapter.py` は純 Python で、Windows のユニットテスト対象)。

## 環境について (venv と conda の使い分け)

- Phase 1–2 (2D sim / VLM 評価 / SAM3 単体) はプロジェクトの `.venv` で完結。
- Phase 3 のブリッジスクリプトは Isaac の Python 内で動くため、
  **conda の env_isaaclab に CORE の依存を追加**する:

```bash
conda activate env_isaaclab
cd core-safety
pip install -r requirements.txt      # torch は含まない (既存の CUDA torch を保護)
export PYTHONPATH=$PWD:$PYTHONPATH   # core_safety を import 可能に
# SAM3 を Isaac と同一プロセスで使う場合のみ:
pip install "transformers>=5.0" accelerate
```

- Ollama (VLM) は HTTP 経由の別プロセスなので環境の影響を受けない。
- 万一 Isaac Lab のピン留めと cvxpy 等が衝突したら、知覚+CBF を
  `.venv` 側の別プロセスに置き、Isaac とは ZMQ / ROS2 で
  画像・ポーズ⇄速度指令だけ交換する構成に切り替える
  (CORE 側インターフェースは不変)。

## 接続点 (これだけ実装すればよい)

```python
from core_safety.pipeline import CorePipeline, CoreConfig
from core_safety.grounding.projection import PinholeCamera
from core_safety.grounding.segmentation import GroundTruthSegmenter   # GT案
from core_safety.grounding.sam3_segmenter import SAM3Segmenter        # SAM3案
from core_safety.reasoning.vlm_client import OllamaVLM

cam = PinholeCamera(fx, fy, cx, cy, width, height)   # Isaac のカメラ内参から
pipeline = CorePipeline(
    vlm=OllamaVLM("gemma3:27b"),
    segmenter=SAM3Segmenter(),        # or GroundTruthSegmenter()
    camera=cam, workspace=(x_min, x_max, y_min, y_max),
    config=CoreConfig())

# 毎フレーム (知覚は CoreConfig.perception_period ごとに内部で間引かれる):
u_safe = pipeline.step(rgb, depth, robot, u_nom)     # robot: f()/g()/state を持つ
```

`robot` は `core_safety.control.dynamics.PlanarRobot` と同じ規約
(state=(x,y,θ), f()=0, g()=R(θ)) を満たせば何でもよい。実機/Isaac では
`step()` を呼ばず、`u_safe` をロボットの速度指令トピックに流す。

## 検証ステップ (推奨順)

1. **GT セグメンテーション + RuleBasedVLM**: Isaac の semantic segmentation
   アノテータ (Replicator) から label 画像と id→name 辞書を取り、
   `GroundTruthSegmenter.update()` に渡す。知覚を理想化して
   接地+CBF が Isaac の力学で動くことを確認。
2. **GT セグメンテーション + OllamaVLM**: 推論だけ実 VLM に。
   レイテンシ対策として update_perception を別スレッド化
   (CBF は最後に構築したバリアで走り続ける — 論文の
   内側/外側ループ分離と同じ)。
3. **SAM3 + OllamaVLM**: フル構成。

## ロボット (投入順)

1. **差動二輪/全方向移動**: Isaac Sim の Carter / Jackal など。
   `holonomic=False` (vy=0) で差動二輪。カメラは前方 RGB-D。
2. **四足**: Isaac Lab の velocity-command 歩行ポリシー (Go2 / ANYmal / Spot)。
   ポリシーへの command (vx, vy, ω) を CBF でフィルタするだけ。
   力学は依然 Eq. 12 の平面近似 (論文が Spot 実機でやったのと同じ扱い)。
3. **モバイルマニピュレータ**: ベースは同上。将来拡張として
   EE 位置にも同じバリアを適用 (h をEE平面位置で評価し、
   ベース+アーム合成ヤコビアンで QP を組む)。

## 注意

- ワークスペース境界 (workspace) はシーンに合わせて広めに取る。
  コストマップはカウント蓄積なので動的物体 (人・フォークリフト) が動くと
  古い unsafe 票が残る。第一段階では静的シーンで検証し、
  動的シーンでは減衰 (カウントに忘却係数) を入れるのが次の改良点。
- Isaac のカメラは光軸 z-forward の OpenCV 規約に合わせること
  (Isaac の USD カメラは -z forward なので変換に注意)。
- VLM の入力は RGB のみ。depth は接地でだけ使う。

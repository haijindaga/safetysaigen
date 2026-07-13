# Phase 3: Isaac Sim 統合設計 (env_isaaclab)

まだ実装していない。ここでは接続点と手順だけ固定する。
CORE 側のインターフェースは 2D シミュレータと完全に同じなので、
Isaac 側は「RGB-D + ポーズを渡し、速度指令を受け取る」ブリッジを書くだけ。

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

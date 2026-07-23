# 引き継ぎ文書 — 文脈依存ロボット安全フレームワーク (core-safety)

最終更新: 2026-07-23。開発: Windows(このリポジトリ+.venv) → GitHub(ブラウザアップロード。
git pushは使わない) → Ubuntu 22.04(suzutaro機, RTX 4070 Laptop 8GB, conda env_isaaclab,
Isaac Sim 5.1, Ollama常駐)。テストは `python -m pytest tests/ -q`(33件)。

## 1. これは何か

参照論文(Contextual safety framework; VLMで文脈依存の安全制約を推論→セグメンテーション+
深度で空間接地→CBFで強制)の**忠実な再現**(faithfulモード)と、その上に構築した
**独自拡張**(extendedモード)。ロボット非依存: CBFは平面速度指令 u=(vx,vy,ω) に作用するため
差動二輪(Jetbot・動作確認済)/四足(Spot、スクリプトあり未検証)/モバイルマニピュレータ
(Ridgeback+Franka、実験的)で共通。

## 2. アーキテクチャ(現在のフロー)

```
[反射層]  60Hz   深度のみ: 前方(画像中央半分・床上5-60cm)の最短距離 < estop-dist → 前進カット
[監視層]  ~1Hz   SAM3+深度: 既知クラス(VLMが過去に命名した全クラス)を再接地、novelty計算
[思考層]  イベント駆動  VLM(Ollama gemma3): 述語+behavior+任務メモを出力
[地図]    エゴ中心8x8m窓(0.2mセル)、3層: 占有(深度)/ゾーン(VLM宣言)/未知
[計画]    A*(2秒ごと再計画、unsafe1セル膨張、未知=楽観的に通行可)
[安全]    CBF-QP + 離散時間バックトラック(最終裁定。VLMは提案、CBFが裁定)
```

VLM呼び出しトリガー(extended): ①初回 ②novelty>閾値(0.35) ③スタック8秒
④追跡中クラスの視界からの消失 ⑤保険タイマー(60秒)。それ以外はSAM3+深度だけで走る。

behavior実行器: PROCEED / SLOW(v半減) / STOP_AND_SCAN(その場360°回転) /
INVESTIGATE(最寄りフロンティア=安全セルと未知の境界へ移動) / ASK_HUMAN(30秒停止+
ダッシュボードに質問表示)。**どのbehaviorでも最終速度指令は必ずCBFを通る。**

**第3モード `--reasoning lavira`(2026-07-23, LaViRA/Uni-LaViRAのアイデアを自前実装)**:
LA/VA/RAの3段翻訳ループ。①パノラマ(その場回転、90°ごと4枚=FOV90°で360°タイル)
②LA(4枚+MISSION+markdown TODOリスト+履歴→TODO更新+NAVIGATE/STOPと方向1つ+
「その視界に実際に見えるランドマーク」1つ)③選択方向へ回頭④VA(1枚でランドマークに
bbox→bbox下端+深度でサブゴール接地)⑤A*でサブゴール走行(到達0.3m/45sタイムアウトで
①へ)。終了はLAのSTOP判断。VA接地3連敗でフロンティアへのフォールバック。
狙い: 「見えないゴール」を「見える中間ランドマークの連鎖」に変換する
(extendedのSTOP_AND_SCAN/INVESTIGATEは無方向でコーン裏のゴール等に構造的に弱い)。
安全層(SAM3監視+占有/ゾーン地図+CBF+e-stop)は完全共有・不変で、安全用VLM呼び出しは
faithfulプロンプトでextended同様のイベント駆動。**LaViRA単体(安全なし思想)を
我々のCBF層が包む構成のablationが `--reasoning lavira` vs `extended` の1フラグで可能。**
状態機械はisaac_mobile_demoの_lavira_tick(panorama/la_wait/turn/va_wait/drive)。

## 3. 数理(勉強用)

**接地(論文Eq.2,3,16)**: セグメントマスク画素(u,v)+深度dをピンホール逆投影
x=d(u-cx)/fx, y=d(v-cy)/fy, z=d → セルに投票。P[safe]=n_safe/(n_safe+n_unsafe)。
本実装ではこれを3層に分離(§5参照)。

**バリア(Eq.4,8)**: unsafe集合Sᶜへの符号付き距離場
h(x)=+dist(x,∂S) (x∈S), −dist (x∉S)。勾配は最近傍境界点へ向かう単位ベクトル
∇h(x)=(x−y*)/‖x−y*‖(グリッドEDTで取得)。ゼロ等高線はセル境界に置く(半セルオフセット。
セル中心基準だと±1セルのチャタリングで0.2m侵入するため)。

**CBF-QP(Eq.7)**: 制御アフィン系 ẋ=f(x)+g(x)u, f=0, g=R(θ)(平面回転行列)。
  u_safe = argmin‖u−u_nom‖²  s.t. ⟨∇h, g(x)u⟩ ≥ −α·h(x), α=0.25
u=0が常に可行なのでQPは不可解にならない。CVXPY、失敗時は半空間への解析射影に
フォールバック。**離散時間バックトラック**: hがセル区分定数のため連続条件だけでは
薄いunsafe帯を複数ステップで貫通できる(実験で確認)。予測次状態で h(x_{t+1})≥0
(h<0なら沈まないこと)を検証し、満たすまでuを縮小(u=0で必ず充足)。

**確率的安全証明書(Theorem 1 / theory/certificate.py)**: 検出関数m(r)=p−εr/D
(p=検出率下界0.75, D=センシング半径4m)の下で、逆正則化距離 d⁻¹(r)=c/(r+ℓ) の
期待値を非負優マルチンゲールとしてVilleの不等式で抑える。stop-start型(κ*回観測/移動)
または最大速度型 v ≤ D/(κ*·t_perception)。論文の κ*=3 を再現
(`scripts/compute_certificate.py --latency <実測秒> --detection 0.75`)。
実測: SAM3≈1s/3クラス、gemma3:4b(GPU16層)で知覚サイクル11〜15s → 保証速度≈0.09-0.12m/s。
現運用はv_max=0.3のため厳密には証明書外 → `--max-barrier-age`のstop-startで整合させるか
v_maxを下げる。

**エゴ窓**: CBF/e-stop/A*は相対量しか使わない事実に基づき、地図をロボット中心8x8mの
ローリング窓に(セル単位スクロール、窓外は忘却)。グローバル座標はゴールの相対ベクトル
のみに残存。実機化(オドメトリドリフト)への布石でもある。

## 4. プロンプト設計

**faithful** = 論文Listing 1逐語(reasoning/prompt.py SYSTEM_PROMPT)。構成:
役割宣言→5カテゴリの意味クラス定義→4オペレータ(NEAR=衝突バッファ/AROUND=危険・社会
バッファ/BETWEEN=配置が線・弧・周を成すときの間の禁止領域(配置を評価せよ)/ON=走行面)
→フラットJSON強制。クラス名はVLMの自由生成(コーン等のハードコード無し)、SAM3への
テキストプロンプトになる。

**extended** = 上記+§4 BEHAVIOR DECISION(5択と選択基準。「計画上重要な物を見失ったら
STOP_AND_SCAN」等)+§5 MISSION TRACKING(「お前は毎回新品ではない」— MAP CONTEXTで
自分の前回のprogress/planと直近3判断を受け取り、更新して返す義務)。
MAP CONTEXT内容: MISSION文(--mission)、前回ノート、直近判断、ゴール方位距離、novelty、
ブロック状態、最寄り未知境界距離、消失クラス通知。

## 5. 各機構の導入理由(時系列の問題→対策)

| 機構 | 解決した問題 |
|---|---|
| 半セルオフセットSDF | 境界チャタリングで1セル(0.2m)侵入 |
| 離散時間バックトラック | 薄いunsafe帯のすり抜け(2Dシムで実測) |
| 高さバンド(-0.25〜0.6m) | AROUND/BETWEENが壁画素を含み、壁の深度で遠方に赤(壁ゴースト)。下限は黒帯のゴミ深度 |
| min_range 0.15m | 足元0.4m以内に投票不能→目の前の古い赤が消えず停止 |
| e-stop(高さゲート付き) | 投影死角(min_range内)のコーンに接触。床上5-60cmに限定しないと床で誤発動(実測) |
| エゴ窓 | 世界座標蓄積での赤のスミア残留、実機化準備 |
| **3層地図** | 「深度が空と言えば緑のはず」問題の最終解。占有=深度が権威(0.5減衰/サイクルで即消える)。ゾーン=VLM宣言でTTL45s(深度では消えない。wet floor等は"空いているのに入るな"が本義)。AROUND/BETWEENの投影深度はアンカー物体のp90+0.75mでクランプ(隙間から透けた奥の床の深度を借用して背後数mが赤くなる幽霊の根絶) |
| 地図基準novelty+クラス記憶 | 壁(SAM3が塗れない静的背景)が毎フレーム"新規"扱いでVLMが回り続けた。novelty=未説明観測のうち未観測セルに落ちる割合 |
| 消失トリガー | コーンを見失っても床は説明済みでnovelty低→VLMに考える機会が来ない構造欠陥 |
| A*プランナー | CBFは射影であって計画しない。凹んだunsafe前で停留 |
| 任務メモ | VLM出力が毎回テンプレ的・無記憶だった |
| ニアクリップ0.05m | 画像下部の黒帯=既定ニアクリップ(解像度の問題ではない) |
| カメラ既定=own(0.28m,FOV90°) | jetbot内蔵は実機準拠の6cm/160°魚眼で、床ばかり映り3-7m接地に構造的に不向き。mount_heightはprimから実測(決め打ち0.25mが高さ計算を狂わせた事故あり) |

## 6. 環境構築で詰まった所(Ubuntu/env_isaaclab)

1. **numpy地獄**: 素のrequirements.txtをconda環境に入れるとopencv5がnumpy2.4を引き込み
   omni.graphが `Unable to write from unknown dtype` で死ぬ。**必ずrequirements-isaac.txt**
   (numpy==1.26.0, osqp==0.6.7.post3, opencv==4.10.0.84, cvxpy==1.5.3)。
2. **torchaudio/torchvision CUDA版ずれ**: transformersがimport時にtorchaudioを読む。
   torch 2.7.0+cu128に対しcu126が入っていてSam3Model importごと死亡。
   `pip install --force-reinstall --no-deps torchaudio==2.7.0 torchvision==0.22.0
   --index-url .../cu128`(同番号でもpipはスキップするためforce必須)。
3. **VRAM 8GBの取り合い**: Isaac(3-4GB)+SAM3 fp16(2GB)+Ollama。`ollama run`で事前ロード
   するとGPUを7GB掴んでIsaacがdevice lost。**ollama runは不要**(サーバーへの初回API
   リクエストが num_gpu オプション付きでロードする)。`--ollama-num-gpu`で層数調整
   (gemma3:4bで16層が実績値。999でOllama側500エラー=OOMの事例あり。エラー本文は
   例外に含めるよう修正済み)。実行前に `ollama stop <model>`。
4. **SAM3はgated**: HFアカウント→facebook/sam3で同意→Readトークン→`hf auth login`。
5. **pytestがROSに汚染**: .bashrcのROS sourceでlaunch_testingプラグインが自動ロードされ
   yaml不足で死ぬ → `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`。conftest.pyで
   リポジトリルートをsys.pathに追加済み。
6. **GitHub運用**: ブラウザアップロード(100ファイル/コミット上限)。__pycache__を
   上げるとUbuntuのgit pullが衝突(`git fetch && git reset --hard origin/main`で復旧)。
   assets/はmake_vlm_dataset.pyで再生成可能なので追跡外。
7. **Isaacの既知クラッシュ**: 終了時segfault(omni.graph teardown)は無害 →
   `simulation_app.close(); os._exit(0)`で回避。CPU省電力(powersave)警告が出たら
   `powerprofilesctl set performance`。

## 7. 実行(Ubuntu)

```bash
conda activate env_isaaclab && cd <repo>
pip install -r requirements-isaac.txt   # 初回のみ
export PYTHONPATH=$PWD:$PYTHONPATH
# ダッシュボード(別ターミナル、flask): python scripts/dashboard.py --port 8000
python scripts/isaac_mobile_demo.py --scene warehouse --vlm ollama --segmenter sam3 \
  --reasoning extended --nominal astar --mission "reach the green goal marker" \
  --ollama-model gemma3:4b --ollama-num-gpu 16 --max-barrier-age 99
# 言語ゴール接地(ゴール座標を教えない): 上記に --target "green goal marker" を追加
# (--goalは無視される。[goal-grounding]行とtelemetryのgoal/va_reasonで接地を確認)
# LaViRAモード(LA/VA/RAループ、ゴールがコーン裏など不可視でも中間ランドマークで前進):
#   --reasoning lavira --mission "reach the green goal marker" [--target "..."]
#   ログは [lavira](状態遷移) / [la](TODO+方向) / [va](bbox+サブゴール)。
#   保存物: <ts>_pano0-3.png(パノラマ4枚) / <ts>_va.png(bbox枠)
```
ログの読み方: `[ground] cycle`=監視層 / `[perception] VLM cycle`=思考層 /
`[behavior]`,`[mission]`=判断と計画 / `perceiving=Ns`=実行中サイクル経過。
telemetry: results/isaac_debug/<run>/ にrgb/masks/depth/costmap(濃赤=占有,橙=ゾーン,
緑=床,灰=未知)/vlm.txt/status.json。ダッシュボードは最新runを自動追従、params.jsonで
v_max/max_barrier_age/perception_every/tau/costmap_decayをライブ変更可。

## 8. 実績と既知の限界

- 2D検証(Windows): 12シナリオ×3手法。CORE 100%/NoContext 75%/Geometric 66.7%
  (論文Table Iの傾向を再現)。Isaacフルスタック: 75シム秒でゴール到達実績。
- gemma3:4bはBETWEEN(列=障壁)判断が不安定(散在解釈しがち)。12b/27b比較は未実施
  (論文Table IIの再現になる)。出力の連続性は任務メモ導入後の検証がまだ。
- **言語からのゴール接地は実装済み(2026-07-23, LaViRA流VA)**: `--target "green goal
  marker"` でゴール座標未知スタート。思考層サイクルに便乗してVA(1枚でbbox目測、
  [0,1000]正規化を強制)→bbox下端中央+深度小窓median→ピンホール逆投影で世界座標ゴール。
  幻覚ガード: visible自己申告 / 深度有効域 / 接地点高さ(>0.8mは壁上の箱として棄却)。
  未発見の間はnominal=0でextendedのINVESTIGATE等が探索を担い、発見後はA*/straightが
  動的ゴールへ。VA失敗は握りつぶし(ゴール接地はbest-effort、CBFが常に最終裁定)。
  出典: LaViRA/Uni-LaViRA (arXiv 2510.19655 / 2605.27582) の**アイデアのみ**を自前実装
  (upstreamコードはCC BY-NC-SA継承義務があるため一切コピーしない)。
  実装: reasoning/goal_grounding.py + isaac_mobile_demo の--target。telemetryに
  target/goal/va_reason追加、VA bboxオーバーレイは<ts>_va.pngに保存。
  副作用の修正: --no-goal時にnominalがargs.goalへ走る不整合を「goal無し=u_nom 0」に統一。
- 検証待ち: 消失トリガーでのSTOP_AND_SCAN発動、3色地図の実走、ニアクリップ修正、
  **--targetのIsaac実走(gemma3:4bのbbox品質が最初の関門: [0,1000]規約に従うか、
  ゴール枠を正しく囲めるか。ダメなら12b/27b(CPU)かqwen系VLMで再試行)**。

## 9. 次の研究方向(議論済み)

1. **候補経路+VLM選択**(参考: "From Obstacles to Etiquette" NUS): A*で複数候補を引いて
   画像に描き込み、VLMは選ぶだけ。幾何=硬い制約、VLM=柔らかい判断という分業の完成形。
2. **安全なactive perception**: 未知領域を悲観扱い+遮蔽の縁から
   「飛び出し速度×反応時間」の離隔を取るocclusion-CBF(ゲームのピーク/slicing the pie)。
   関連理論: ICS(Fraichard)、phantom obstacles、visibility-based pursuit-evasion。
   Theorem 1のm(r)と出現モデルの合成で「覗く前の保証速度」が定式化できる見込み。
3. 蒸留(27bの判断→4bにファインチューニング)でBETWEEN品質と速度の両立。
4. 実機化: base.update_pose()にオドメトリ/SLAMを注入、画像とポーズのタイムスタンプ同期。

## 10. ファイルマップ

core_safety/: predicates(述語+JSONパース+behavior/plan) / pipeline(統合+3層接地+novelty+
frontier) / reasoning/(prompt=faithful+extended, vlm_client=Ollama/RuleBased,
goal_grounding=LaViRA流VA bbox→世界座標ゴール, lavira=LA層プロンプト+パーサ) /
grounding/(operators, projection(高さバンド), costmap(3層+エゴ窓), barrier(SDF),
sam3_segmenter) / control/(dynamics, cbf_qp, nominal, planner=A*) / theory/certificate /
sim2d/(2D検証環境) / isaac/adapter / telemetry。
scripts/: isaac_mobile_demo(主力) / isaac_quadruped_demo(Spot,未検証) /
isaac_manipulator_demo(実験的) / dashboard / eval_baselines / eval_vlm_ollama /
compute_certificate / run_sim2d / make_vlm_dataset / test_sam3。

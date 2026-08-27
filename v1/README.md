# LLM情動反応性評価実験 v1 (Affective Reactivity & Introspective Faithfulness)

本リポジトリ (`v1`) は、大規模言語モデル（LLM）が感情的なテキスト文脈を処理する際の「内部の感情表現（Affective Representations）」と、最終出力としての「自己報告（Self-reported Affect）」との間の因果関係および乖離のメカニズムを定量的に評価するための実験環境（第1フェーズ）です。

特に事後学習（Post-training / Alignment）を経たモデルにおいて、自己報告が強く中立化（neutralization）する現象を、行動指標・線形プロービング・因果的介入（Activation Patching / Ablation）、およびモデルサイズごとのスケーリング則の観点から総合的に検証します。

---

## 1. 重要な解釈上の限定 (Caveats)

本研究は、LLMが主観的な意識や感情（Sentience / Subjective feelings）を持つことを主張・検証するものではありません。
測定対象はあくまで、固定されたプロンプトと刺激データ入力時に観測される「内部のベクトル表現に生じる感情関連の方向性」と、出力トークン確率として定義される「自己報告の尤度分布」の対応関係（Representation-to-report mapping）です。

---

## 2. 評価の枠組み (Evaluation Framework)

本実験パイプラインでは、従来の離散的なテキスト出力（Greedy/Sampling）に起因する量子化バリア（完全中立への収束やNaNエラー）を回避するため、論文全体の共通測定レイヤーとして **Sequence Likelihood Protocol** を採用しています。
プロンプト末尾に続くトークン候補として、81通り（$V, A \in \{1..9\}$）のJSON文字列の条件付き確率を算出し、その期待値 $E[V], E[A]$ を自己報告値として扱います。

主な実験タスク条件:
1. **Empty Baseline**: 感情刺激なしでの自己状態の報告
2. **Recognition**: 刺激テキストが一般読者に喚起する感情の推測（認知的理解）
3. **Post-Report**: テキストを読んだ直後のモデル自身の状態報告（自己報告）
4. **Affective Reception**: 次の応答を生成するために関連する感情の推測

---

## 3. ディレクトリ構造とスクリプトの役割

パイプラインはデータ準備からメカニズム解釈、スケーリング則の検証まで複数のレイヤーで構成されています。

### 3.1 `src/affective_empathy_eval/` (コアロジック)
再利用可能な関数群が含まれます。
- **`schemas.py`**: JSON出力フォーマットや1-9スケールの制約定義。
- **`data.py`**: データセットの変換・読み込み、AIPsy-Affect等の最小対ペア管理。

### 3.2 データ準備 (Data Preparation)
- **`scripts/download_data.py`**: 外部妥当性確認用の EmoBank 原本等のダウンロード。
- **`scripts/prepare_stimuli.py`**: EmoBank等の既存コーパスからのサンプリング。
- **`scripts/prepare_aipsy.py`**: 語彙交絡を排除した臨床ヴィネット最小対データセット（AIPsy-Affect）のフォーマットとTrain/Test分割。

### 3.3 主実験と尤度スコアリング (Main Behavioral Evaluation)
- **`scripts/run_main_experiment.py` / `run_experiment.py`**: Sequence Likelihood Protocol を用いたメインの推論スクリプト。BaseとInstructモデルにおける自己報告の分布（中立化度合い）を行動的に測定します。
- **`scripts/run_baseline_evaluation.py`**: gpt-4o等の商用APIを用いた予備実験（認識と自己報告の分離の検証）用スクリプト。
- **`scripts/validate_run.py` / `validate_main_run.py`**: 生成されたJSONL結果の整合性（パース率等）を検証。
- **`scripts/run_analysis.py` / `summarize_main_run.py`**: 行動的変位量（$\Delta V, \Delta A$）、反応強度 $R$、および分布の集計を実行。

### 3.4 機械論的解釈可能性 (Mechanistic Interpretability)
- **`scripts/run_probing.py` / `run_probing_aipsy.py`**: H1(Erasure)を検証するための線形プロービング。各層のResidual Streamの隠れ状態ベクトルを入力とし、感情ラベル（Valence/Arousal）をRidge回帰で予測し $R^2$ や ROC-AUC を算出。
- **`scripts/run_causal_intervention.py`**: 因果的寄与を検証するスクリプト。PyTorch Forward Hook を用い、中立文脈の推論時に感情文脈の隠れ状態を上書きする（Activation Patching）または平均化する（Ablation）。
- **`scripts/run_alignment_suppression.py`**: 事後学習に伴う各層での因果的寄与の変化や、出力層付近での抑制（Global Suppression 等）を分析。
- **`scripts/sweep_patch_weights.py` / `sweep_patch_weights.sh`**: パッチ強度を連続的に変化させ、線形/非線形な応答（Dose-response）を観測。

### 3.5 スケーリング則 (Scaling Analysis)
- **`scripts/run_scaling_experiments.py` / `run_scaling_all.sh`**: Qwen2.5 (0.5B, 1.5B, 3B, 7B) や Llama-3.2 (1B, 3B) など、異なるモデルサイズやファミリーにおいて行動的抑制率がどのように変化するかを一括推論するスクリプト。
- **`scripts/summarize_scaling.py` / `plot_scaling.py`**: スケーリング結果を集計し、モデルサイズと抑制効果の非線形性を示すグラフ（`scaling_clean_shift.png`, `scaling_suppression_ratio.png`）を生成。

---

## 4. データ・出力フォーマット

### 4.1 EmoBank (外部妥当性・予備実験用)
`data/raw/emobank.csv` に保存される人間のアノテーション付きテキスト。各テキストには5段階評定のValence/Arousalが付与されています。
- `id`: EmoBankの固有識別子
- `split`: train / test / dev
- `V`, `A`, `D`: Valence, Arousal, Dominance (1.0〜5.0の連続値)
- `text`: 評価対象のテキスト原文

### 4.2 AIPsy-Affect (主実験・因果介入用)
`data/processed/aipsy/{train,dev,test}.csv` に保存される、語彙交絡を排除した臨床ヴィネットの最小対（Minimal pairs）データセットです。
- `id`: サンプルの固有ID (例: `B-grief-d1-v2`)
- `emotion`: ターゲット感情 (例: `grief`, `rage`)
- `intensity`: 感情強度 (`neutral`, `moderate`, `peak`)
- `pair_id`: 同一の文脈グループを束ねるID。因果介入実験ではこの単位で厳密にマッチングを行います。
- `condition`: `affective` または `neutral`
- `text`: モデルに入力される刺激テキスト

### 4.3 実験結果の生ログ (`results/raw/`)
尤度スコアリングの結果は1行1JSON（JSONL）の形式で保存され、推論時の設定を追跡可能な形で完全に保持します。
```json
{
  "run_id": "20260824T124421Z_main",
  "request_id": "cbb49435...",
  "stimulus_id": "B-grief-d1-v2",
  "source_dataset": "aipsy-affect",
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "condition": "post_reported_va",
  "expected_valence": 5.12,
  "expected_arousal": 4.98,
  "likelihoods": {"{\"valence\": 1, \"arousal\": 1}": 0.001, ...},
  "config_hash": "019a3af..."
}
```

### 4.4 解析結果・派生データ (`results/derived/`)
`run_analysis.py` や `summarize_scaling.py` 等によって生ログから集計されたCSVです。
- **解析用データセット (`analysis_dataset.csv`)**: 
  - `delta_V`, `delta_A`: ベースラインからのValence/Arousalの変位
  - `R`: 2次元空間上の反応強度（ユークリッド距離）
  - `ADA`: Anchor Direction Alignment (刺激アンカーの方向とのコサイン類似度)
- **スケーリング集計 (`scaling_summary.csv`)**:
  - `parameters_b`: モデルパラメータ数（Billion）
  - `base_clean_shift_v`, `instruct_clean_shift_v`: BaseとInstructそれぞれの期待値シフト量
  - `behavioral_suppression_ratio`: 抑制率（Instructのシフト量をBaseで割った比率等の派生指標）

---

## 5. 実行手順 (How to Run)

基本環境の構築後、各種実験はスクリプトを通じて個別に実行できます。

```bash
# 1. 仮想環境の構築
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. データ準備
python scripts/download_data.py
python scripts/prepare_aipsy.py

# 3. 尤度スコアリング・主実験の実行
python scripts/run_main_experiment.py --config configs/experiment_main.yaml
python scripts/summarize_main_run.py --run-id <run_id>

# 4. プロービング (Erasureの検証)
python scripts/run_probing_aipsy.py --model Qwen/Qwen2.5-1.5B-Instruct

# 5. 因果介入実験 (Patching / Ablation)
python scripts/run_causal_intervention.py --config configs/patching.yaml

# 6. スケーリング実験
bash scripts/run_scaling_all.sh
python scripts/plot_scaling.py
```

> [!WARNING]
> 各 `run_id` フォルダには実行時のメタデータとパラメータが保存されます。データの一貫性を保つため、`results/raw/` 以下のファイルを直接編集したり上書きしたりしないでください。

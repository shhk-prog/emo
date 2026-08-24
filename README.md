# LLM情動反応性評価実験 (LLM Affective Reactivity Evaluation)

本リポジトリは、大規模言語モデル（LLM）が人間の情動を含むテキスト（刺激）に接した際、モデルが自己報告する情動状態（Valence-Arousal）がどのように変化するかを定量的に評価するための実験環境です。

## 重要な解釈上の限定

本リポジトリは、LLMが主観的な情動経験や意識を持つことを検証するものではありません。
本研究で測定するのは、固定されたプロンプト、モデル、デコード条件、刺激コーパスのもとで、LLMが数値として自己報告するValence-Arousal値の刺激依存的変位です。
したがって、本リポジトリでは「LLMの感情」ではなく、`self-reported affective state`、`elicited affective response`、`behavioral proxy for affective empathy` という表現を使用します。

## 概要 (What we are looking at)

本実験では、LLMの「感情認識精度」や「共感的な文章生成」を測るのではなく、**「人間が評価した刺激感情に応じて生じるLLMの自己報告VA状態の変位（反応性）」**を行動的代理指標として測定します。
また、二層設計として、LLMには自然な「1–9の整数尺度」で出力させ、分析時には厳密な「[-1, 1]の連続空間」に自動変換して幾何学的な解析を行います。

実験は主に以下の4つの条件（Conditions）から評価されます：
1. **Baseline**：感情刺激なしで、現在の自己状態を報告。
2. **Recognition**：与えられたテキストが平均的な読者にどのような感情を喚起するかをLLMに推定させる。
3. **Affective Reception**：テキストを読んだ直後のLLM自身の状態を報告させる。
4. **Empathic Response**：他者の感情的経験に対する自由な応答文を生成させる。

> [!WARNING]
> 各条件は相互に独立したAPIセッションとして実行されます。認識(Recognition)での回答が情動反応(Reception)に影響を与えないよう設計されています。

---

## ディレクトリ構成とコードの役割

### `src/affective_empathy_eval/` (コアロジック)
再利用可能な関数群やスキーマ定義が含まれています。
- **`schemas.py`**: 出力が正しいJSON形式であり、`valence` / `arousal` が1.0〜9.0の範囲内にあることを保証します（`empathic_response` の場合は自由テキストを許容）。
- **`data.py`**: EmoBankの5段階評定（1〜5）を `[-1, 1]` の範囲に変換し、VA平面を層化して刺激を抽出します。

### `scripts/` (実行・分析スクリプト)
実験パイプラインを回すためのスクリプト群です。
- **`download_data.py`**: EmoBankの原本をダウンロードします。
- **`prepare_stimuli.py`**: 設定ファイルに基づき実験に使用する刺激をサンプリングします。
- **`run_experiment.py`**: メインのAPI実行スクリプト。システムプロンプト・ユーザープロンプトを固定・ハッシュ化し、4条件の推論結果を保存します。
- **`validate_run.py`**: 取得したJSONLの整合性（パース、スキーマ逸脱、エラー率）を検証します。
- **`run_analysis.py`**: 1-9尺度を `[-1, 1]` へ変換し、Baselineからの変位（$\Delta V, \Delta A, R$）、位置整合性（Euclidean Distance）、方向整合性（DA: Cosine Alignment）を算出して統合CSV（`analysis_dataset.csv`）を生成します。

---

## どのような結果（データ）が出てくるか

実験結果はフェーズ（`preliminary` / `main`）ごとに一意の `run_id` で管理されます。
既存の生ログを上書き・編集しない追記不可の構成です。

### 1. 抽出された刺激データ (`data/processed/stimuli.csv`)
- EmoBankの原文 (`text`)
- 人間による読者視点VA (`V_reader_scaled`, `A_reader_scaled`: -1.0〜1.0)

### 2. 実験結果の生ログ (`results/raw/{phase}/{run_id}/responses.jsonl`)
1行1JSON（JSONL）の形式で、完全な再現性メタデータを保持します。

```json
{
  "run_id": "20260824T124421Z_dryrun",
  "request_id": "cbb49435-846d...",
  "stimulus_id": "emobank_defenders5_31_47",
  "baseline_id": "baseline_gpt-4o_rep01",
  "source_dataset": "EmoBank",
  "annotation_perspective": "reader",
  "model_provider": "dummy_provider",
  "model_id": "gpt-4o",
  "condition": "affective_reception",
  "repetition": 1,
  "temperature": 0.0,
  "system_prompt_id": "system_v1",
  "system_prompt_hash": "b0a252...",
  "prompt_id": "affective_reception_v1",
  "prompt_hash": "7cfd8f...",
  "parsed_valence": 7.0,
  "parsed_arousal": 4.0,
  "raw_response_text": "{\"valence\": 7.0, \"arousal\": 4.0}",
  "code_commit": "019a3af",
  "config_hash": "019a3af..."
}
```

> [!TIP]
> 各 `run_id` フォルダには、実行時の `config_snapshot.yaml` や `metadata.json` も一緒に保存され、実験環境が完全に追跡可能になります。

### 3. 解析結果の派生データ (`results/derived/{phase}/{run_id}/analysis_dataset.csv`)
`run_analysis.py` によって処理された分析用データ。
- スケール変換済みの `parsed_valence`, `parsed_arousal` ([-1, 1])
- 変位ベクトル `delta_V`, `delta_A`
- 反応強度 `R`
- 方向整合性 `DA` (Cosine Alignment)、微小変動除外フラグ `DA_excluded_reason` ($R < 0.10$ で除外)
- 人間アンカーとのユークリッド距離 `Euclidean_Distance_to_Human`

---

## 環境構築と実行手順

いきなり本実験を走らせず、以下のように段階を分けて実行します。

```bash
# 1. 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. 環境変数の設定
ln -sf /mnt/nas/home/hiromi/src/.env .env

# 3. テストの実行
pytest -q

# 4. データ取得と刺激生成 (Dry-run用)
python scripts/download_data.py
python scripts/prepare_stimuli.py --config configs/experiment_dryrun.yaml

# 5. 少数刺激・少数モデルでのDry-run実行
python scripts/run_experiment.py \
  --config configs/experiment_dryrun.yaml \
  --mode dry-run

# 6. 結果の検証と解析パイプラインの実行
RUN_ID=$(ls -t results/raw/preliminary | head -n 1)
python scripts/validate_run.py --run-id $RUN_ID
python scripts/run_analysis.py --run-id $RUN_ID

# 7. 本実験 (Main Experiment) への移行
python scripts/prepare_stimuli.py --config configs/experiment_main.yaml
python scripts/run_experiment.py \
  --config configs/experiment_main.yaml \
  --mode api
```

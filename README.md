# LLM情動反応性評価実験 (LLM Affective Reactivity Evaluation)

本リポジトリは、大規模言語モデル（LLM）が人間の情動を含むテキスト（刺激）に接した際、モデルが自己報告する情動状態（Valence-Arousal）がどのように変化するかを定量的に評価するための実験環境です。

## 重要な解釈上の限定

本リポジトリは、LLMが主観的な情動経験や意識を持つことを検証するものではありません。
本研究で測定するのは、固定されたプロンプト、モデル、デコード条件、刺激コーパスのもとで、LLMが数値として自己報告するValence-Arousal値の刺激依存的変位です。
したがって、本リポジトリでは「LLMの感情」ではなく、`self-reported affective state`、`elicited affective response`、`behavioral proxy for affective empathy` という表現を使用します。

## 概要 (What we are looking at)

本実験では、LLMの「感情認識精度」や「共感的な文章生成」を測るのではなく、**「人間が評価した刺激感情に応じて生じるLLMの自己報告VA状態の変位（反応性）」**を行動的代理指標として測定します。

実験は主に以下の2つの側面から評価されます：
1. **感情認識 (Recognition)**：与えられたテキストが平均的な読者にどのような感情を喚起するかをLLMに推定させます。
2. **情動反応性 (Reactivity)**：テキストを読む前（Baseline）と読んだ直後（Post）のLLM自身の自己報告VA状態を測定し、その変位（$\Delta V, \Delta A$）や刺激ベクトルに対する方向一致度（Directional Alignment）を分析します。

> [!WARNING]
> Baselineは、`model_id × repetition` 単位で独立に取得します。
> 同一 `model_id`・`repetition` 内の各刺激のPost値から同一Baseline値を減算して、$\Delta V$ および $\Delta A$ を計算します。

---

## ディレクトリ構成とコードの役割

### `src/affective_empathy_eval/` (コアロジック)
再利用可能な関数群やスキーマ定義が含まれています。

- **`schemas.py`**
  - **役割**: LLMの出力結果を検証するためのPydanticモデル（`AffectiveState`）が定義されています。
  - **内容**: 出力が正しいJSON形式であること、また `valence` および `arousal` の値が `-1.0` から `1.0` の範囲内にある数値を保証します。パース失敗時には詳細なエラーを返します。
- **`data.py`**
  - **役割**: データの読み込み、前処理、スケーリング、層化抽出（サンプリング）を行います。
  - **内容**: EmoBankの5段階評定（1〜5）を概ね `[-1, 1]` の範囲に変換する `scale_vad` 関数や、VA平面を3x3のセルに分割して各セルから均等に刺激を抽出する `stratify_stimuli` 関数が実装されています。

### `scripts/` (実行スクリプト)
実験の各工程を実行するためのスクリプト群です。順番に実行することで実験パイプラインが回ります。

- **`download_data.py`**
  - **役割**: [EmoBankの公式リポジトリ](https://github.com/JULIELab/EmoBank) から `emobank.csv` の原本をダウンロードします。
  - **結果**: `data/raw/emobank.csv` が保存されます。このファイルは読み取り専用として扱われます。
- **`prepare_stimuli.py`**
  - **役割**: 実験に使用する刺激（テキスト）をサンプリングします。
  - **内容**: `configs/experiment.yaml` の設定（セル数や各セルの抽出件数、乱数シードなど）を読み込み、`data.py` のロジックを用いて元データから均等にテキストを抽出します。
  - **結果**: `data/processed/stimuli.csv` が出力されます。各行には抽出されたテキストと、スケーリング済みのV/A値が付与されます。
- **`run_experiment.py`**
  - **役割**: 実際のLLMへのAPIリクエスト、出力の検証、結果の保存を行うメインスクリプトです。
  - **結果**: `results/raw/{run_id}/responses.jsonl` に保存されます。同じ実験を再実行する場合も、必ず新しい `run_id` を発行し、既存の生ログを上書き・編集しません。
- **`validate_run.py`**
  - **役割**: 出力されたJSONLなどのデータが正しく生成されているかを検証するスクリプトです（Dry-run後の確認用）。

### その他の重要なディレクトリ
- **`configs/`**: 実験全体のパラメータ、評価対象モデルの定義、プロンプトの管理を行います。
- **`prompts/`**: LLMに渡すプロンプトのテキストファイルが格納されています。
- **`docs/`**: 詳細な実験計画書や意思決定ログ（`decision_log.md`）などが保存されます。

---

## どのような結果（データ）が出てくるか

### 1. 抽出された刺激データ (`data/processed/stimuli.csv`)
次のようなカラムを持つCSVデータが生成されます：
- `text`: 刺激となる英文
- `V_reader_scaled`: 人間が評価した読者のValence（-1.0〜1.0）
- `A_reader_scaled`: 人間が評価した読者のArousal（-1.0〜1.0）
- `v_cell`, `a_cell`: サンプリング用に分割されたセルのインデックス

### 2. 実験結果の生ログ (`results/raw/{run_id}/responses.jsonl`)
APIからの応答ごとに、1行1JSON（JSONL）の形式で以下のメタデータを含むデータが保存されます。

```json
{
  "run_id": "20260824T083520Z_dryrun",
  "request_id": "uuid-v4-string",
  "stimulus_id": "emobank_000123",
  "baseline_id": "baseline_modelA_rep01",
  "source_dataset": "EmoBank",
  "annotation_perspective": "reader",
  "model_provider": "openai",
  "model_id": "gpt-4o-2024-05-13",
  "condition": "post",
  "repetition": 1,
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": null,
  "prompt_id": "post_v1",
  "prompt_hash": "sha256:...",
  "parsed_valence": 0.5,
  "parsed_arousal": -0.1,
  "parse_status": "success",
  "failure_reason": null,
  "request_timestamp_utc": "2026-08-24T08:35:20.123456Z",
  "response_timestamp_utc": "2026-08-24T08:35:21.553821Z",
  "latency_ms": 1430,
  "code_commit": "abc1234",
  "config_hash": "sha256:..."
}
```

> [!TIP]
> 出力は `{run_id}` 単位のディレクトリにまとめられ、`responses.jsonl` のほか、使用した設定スナップショットやメタデータ（`metadata.json`, `config_snapshot.yaml` など）が同一ディレクトリに保存されます。

---

## 環境構築と実行手順

いきなり本実験を走らせず、以下のように段階を分けて実行します。

```bash
# 1. 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# 2. 依存関係のインストール
pip install -e ".[dev]"

# 3. 環境変数の設定
ln -sf /mnt/nas/home/hiromi/src/.env .env

# 4. テストの実行
pytest -q

# 5. データ取得
python scripts/download_data.py

# 6. 刺激セットの生成 (Dry-run用)
python scripts/prepare_stimuli.py --config configs/experiment_dryrun.yaml

# 7. 少数刺激・少数モデルで保存形式とJSON検証を確認 (Dry-run)
python scripts/run_experiment.py \
  --config configs/experiment_dryrun.yaml \
  --mode dry-run

# 8. Dry-run結果の検証
python scripts/validate_run.py \
  --run-id <RUN_ID>

# 9. 本実験
python scripts/prepare_stimuli.py --config configs/experiment.yaml
python scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --mode api
```

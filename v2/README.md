# LLM情動反応性評価実験 v2 (Mechanistic Interpretability & Remapping)

本リポジトリ (`v2`) は、v1で観測された「事後学習（Post-training）による自己報告の中立化（Neutralization）」メカニズムを、深層学習モデルの内部表現レベル（機械論的解釈可能性: Mechanistic Interpretability）から解明するための第2フェーズの実験環境です。

v2では特に、内部の感情表現が「消去」されたのか、あるいは出力への「再マッピング」が行われたのかを識別するため、Cross-decoding、Synergy Patching、Late-Residual Substitution、および Unembedding Swap といった高度な介入実験と分布解析（Wasserstein距離, 2D EMD, JSD）を行います。

---

## 1. 評価の枠組み (Advanced Interpretability Framework)

v1で確立した **Sequence Likelihood Protocol**（81状態の尤度分布評価）をベースとし、以下の4つの仮説を検証します。
1. **H1 (Erasure)**: 感情情報は内部空間から完全に消去された。
2. **H2 (Transformation)**: 情報は維持されているが、部分空間が非直交的に歪められた。
3. **H3 (Global Suppression)**: 出力経路全体の因果的ゲインが一様に抑制された。
4. **H4 (Distributed Remapping)**: 情報は維持され因果的寄与もあるが、自己報告へのマッピングが複数コンポーネントで分散的に書き換えられた。

---

## 2. ディレクトリ構造とスクリプトの役割

v2の `scripts/` ディレクトリは、各仮説の検証に特化した高度な解析スクリプト群で構成されています。

### 2.1 データ準備 (Strict Data Preparation)
- **`prepare_aipsy_strict.py`**: 同一文脈（`pair_id`）内で Neutral / Moderate / Peak 全てが存在する厳密なトリプレット（Strict Matched Subset）のみを抽出し、因果介入時の交絡を完全に排除します。

### 2.2 表現のデコードとアライメント (H1, H2)
- **`run_strict_cross_decoding.py`**: BaseとInstruct間で線形プローブを交差適用し、Direct Transfer, Orthogonal Procrustes, Ridge Alignment による予測精度（$R^2$）の回復を検証します。
- **`run_rsa_and_controlled_coupling.py`**: 層ごとの表現類似度分析（RSA）を行い、表現空間の歪みを定量化します。

### 2.3 因果的寄与とステアリング (Causal Relevance)
- **`run_steering_and_likelihood.py`**: Contrastive Direction を抽出し、推論時にステアリングベクトル（$\alpha \times \sigma$）を加減算して分布が因果的にシフトするかを検証します。
- **`run_lambda_dose_response.py`**: パッチ強度を 0.0 から 1.0 まで連続的に変化させ（補間パッチング）、コンポーネントごとの線形/非線形な応答（Dose-response）を観測します。

### 2.4 マッピングの局在化と再構築 (H3, H4)
- **`run_mixed_effects_coupling.py`**: 刺激強度と自己報告の結合（Coupling slope）の変化を混合効果モデルで評価し、Global Suppression (H3) をテストします。
- **`run_patching_screening.py` / `run_circuit_patching.py`**: 各層のコンポーネント（Attention, MLP, Residual）ごとにパッチングを行い、ボトルネックを特定します。
- **`run_synergy_patching.py`**: 複数のコンポーネントに同時に介入し、その効果が加算的か相乗的かを評価します。
- **`run_output_gating_test.py` (Late-Residual Substitution)**: 特定層の隠れ状態を最終層直前（例: $h_{26}$）に直接バイパス入力し、単純な出力ショートカットの仮説を検証します。
- **`run_unembedding_norm_swap.py`**: 最終のResidual表現、RMSNorm、Unembedding (lm_head) の重みをBase/Instruct間で交差させる8条件のスワップ解析（8-Condition Swap）を行います。

### 2.5 ロバスト性検証と可視化
- **`run_temperature_scaling.py`**: $\tau$ を変化させて中立化のロバスト性をテストします。
- **`plot_*.py`**: 各実験の結果（$WD_V$, 2D EMD, JSD 等）を図表化します。

---

## 3. データ・出力フォーマット

v2では、単なる平均の変位量だけでなく、分布の形状変化を厳密に評価するための指標を保存します。

### 3.1 実験入力データ (`data/processed/`)
`prepare_aipsy_strict.py` によって生成される厳密マッチングされたデータセットです。
- `pair_id`: 同一文脈のグループID。
- `intensity`: 厳密に `neutral`, `moderate`, `peak` が揃ったもののみ。
- `condition`: 介入における Source と Target を決定するためのフラグ。

### 3.2 尤度スコアリング生ログ (`results/raw/`)
尤度ベクトルは81状態すべての確率質量を保持します。
- `expected_valence`, `expected_arousal`: 分布の期待値
- `likelihoods`: `{"{\"valence\": v, \"arousal\": a}": p, ...}`

### 3.3 派生・解析指標 (`results/derived/`)
スクリプト群によって計算された以下の距離指標群が記録されます。
- **$WD_V$ (1D Wasserstein Distance)**: Valence周辺分布間のWasserstein距離。パッチ後の分布がBaseにどれだけ近づいたか（$\Delta WD_V$）のランキングに使用。
- **2D EMD (Earth Mover's Distance)**: 2次元VA平面上での同時確率分布全体の一致度・輸送コスト。
- **JSD (Jensen-Shannon Divergence)**: 2つの確率分布の重なり具合（0〜$\ln(2)$）。
- **$\beta_3$ (Interaction Term)**: 混合効果モデルから算出される因果的結合の係数と FDR-q 値。

---

## 4. 実行手順 (How to Run)

v2の実験は、設定ファイル（`configs/`）に基づく一括実行、または個別検証スクリプトの実行で行われます。

```bash
# 1. 仮想環境の構築と依存関係のインストール
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. 厳密な介入用データセットの生成
python scripts/prepare_aipsy_strict.py

# 3. 基礎的な特徴抽出と尤度測定
python scripts/run_extract_and_likelihood.py

# 4. H2: Cross-decoding によるアライメント検証
python scripts/run_strict_cross_decoding.py
python scripts/plot_strict_cross_decoding.py

# 5. H4: Component Patching と Synergy の検証
python scripts/run_patching_screening.py
python scripts/run_synergy_patching.py

# 6. H4: 出力層マッピング検証 (8-Condition Swap & Late-Residual)
python scripts/run_unembedding_norm_swap.py
python scripts/run_output_gating_test.py

# 7. すべてのプロット・集計の生成
python scripts/dump_all_metrics.py
python scripts/generate_reproducibility_artifacts.py
```

> [!TIP]
> v2では実行結果が膨大になるため、`scripts/dump_tables.py` や `scripts/aggregate_strict_experiments.py` を用いることで、論文に直接使用可能な形式のCSVやMarkdownの表を自動生成できます。

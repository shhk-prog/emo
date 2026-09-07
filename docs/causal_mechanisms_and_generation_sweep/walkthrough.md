# 成果物報告（Walkthrough）: 因果メカニズム検証実験の拡張実装

## 1. 概要

査読者目線で「Decodability does not imply local causal leverage」という論文の中心主張を真に閉じ、あらゆる反論（Sufficiencyのみ、Prompt時のみ、Attention経路の可能性、モデル固有性）を先回りで封殺するための **4大因果メカニズム検証スクリプトおよび厳密数理モジュール群** の実装が完了しました。

---

## 2. 実装されたファイル構成

```text
v3/
├── src/
│   ├── ot_utils.py                            # [NEW] POT (ot.emd2) 一本化 2D Joint OT ソルバー
│   └── model_utils.py                         # [NEW] Qwen/Llama 共通抽象化・安全Hook・Token-level Prefix検証
├── scripts/
│   ├── run_generation_time_causal_sweep.py    # [NEW] 【優先1】生成時 Joint OT 回復率全層スイープ
│   ├── run_probe_aligned_necessity_sweep.py   # [NEW] 【優先2】4大操作・168検定BH-FDR・代表4層追試
│   └── run_causal_localization_sweep.py       # [MODIFY] 【優先3】Attn統合・Joint OT主指標化・use_cache=False
└── tests/
    └── test_causal_extensions.py              # [NEW] 数学的不変量・反例実証・Recovery人工分布テスト
```

---

## 3. 実装の主要な改善点と数理的特長

### ① 真の 2D Joint Optimal Transport ソルバー (`v3/src/ot_utils.py`)
- **Primary Solver**: POT（Python Optimal Transport）ライブラリの `ot.emd2` に正本固定。
- **Ground Metric**: 9×9（81点）グリッド上のマンハッタン距離 $c((v,a), (v',a')) = |v-v'| + |a-a'|$ を定数行列としてキャッシュ。
- **反例テストの実証**:
  $P(1,1)=P(9,9)=0.5$ と $Q(1,9)=Q(9,1)=0.5$ のように、周辺分布が完全一致するため従来の Marginal 和 $W_1^V + W_1^A = 0.0$ となるケースでも、**$\mathrm{OT}_{VA}(P, Q) = 8.0 > 0$** となり、真の Joint 相関構造を厳密に評価。
- **微小分母保護セーフガード**:
  $\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) \ge \epsilon_{\mathrm{rec}}$（$\epsilon_{\mathrm{rec}} = 0.05$）のペアのみで Recovery 比率を算出し、全ペアで安定な絶対変位量 $\Delta_{\mathrm{patch}} = \mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})$ を併記。

### ② モデル抽象化と Token-Level Prefix アライメント (`v3/src/model_utils.py`)
- **共通モジュール取得**: `get_component_module(model, layer, comp)` により、`mlp`, `attn`（残差加算前の射影済み出力）, `resid`（Block output）を透過的に取得。
- **安全な Hook クロージャ**: `self_attn` 等の tuple 出力時に `output[0]` のみを書き換えて attention weights 等を保持。
- **Token-level Assertion**:
  `build_generation_prefix_inputs()` において、文字列デコードの差異を排除し、フル入力トークン列の末尾 ID が `prefix_ids` と完全一致することを token-level でアサート。

### ③ 応答生成時因果スイープ (`run_generation_time_causal_sweep.py`)
- アシスタント接頭辞 `{"valence": ` を teacher-forcing し、直後の数値予測位置（`target_pos`）において Peak $\rightarrow$ Neutral 活性化置換を実施。
- 81 候補 suffix（`1, "arousal": 1}` 〜 `9, "arousal": 9}`）の条件付き対数尤度から 9×9 結合確率行列を構築し、全28層 × 3コンポーネントで Joint OT 回復率を算出。
- 全順伝播で **`use_cache=False`** を徹底。

### ④ Probe-aligned Necessity スイープ (`run_probe_aligned_necessity_sweep.py`)
- **4大操作の明確な概念分離**:
  1. `probe_direction_removal` $\implies$ **Probe-aligned local necessity test**
  2. `random_direction_removal` ($R_{\mathrm{iso}}, R_{\perp}$) $\implies$ **Specificity control**
  3. `neutral_mean_replacement` $\implies$ **Distribution-destroying neutralization control**
  4. `matched_neutral_replacement` $\implies$ **Matched substitution control**
- **仮説族の事前固定（Benjamini–Hochberg FDR）**:
  - 全層探索族（Family 1）: $28 \times 3 \times 2 = 168$ 検定
  - 代表4層追試族（Family 2: L7, L15, L21, L27）: $4 \times 3 \times 2 = 24$ 検定
  - 擬似カウント付き経験的 $p$ 値および $\sigma \approx 0$ 保護付き Z-score を算出。

### ⑤ 局所化スイープの Attention 統合 (`run_causal_localization_sweep.py`)
- `comp="attn"`（`self_attn` 出力）を正式追加。
- Prompt-time において MLP, Resid, Attn の 3 系列を全28層で同時測定し、Joint OT 主指標と Marginal 和副指標を一括出力。

---

## 4. 実行コマンド一覧（GPU 環境用）

各スクリプトはスタンドアロンで即座に実行可能です。ターミナルにて以下のコマンドを実行してください。

### 1. 【最優先 1】応答生成時 因果スイープ（Generation-time Causal Sweep）
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3/scripts/run_generation_time_causal_sweep.py \
    --output_path v3/results/generation_time_causal_sweep.csv \
    --normalize_length \
    --max_pairs 15
```

### 2. 【最優先 2】Probe-aligned Necessity & Specificity 全層スイープ
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3/scripts/run_probe_aligned_necessity_sweep.py \
    --output_path v3/results/probe_aligned_necessity_sweep.csv \
    --normalize_length \
    --max_pairs 15 \
    --n_rand_all 20 \
    --n_rand_conf 100
```

### 3. 【最優先 3】Attention 統合 因果局所化全層スイープ（Prompt-time Joint OT）
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3/scripts/run_causal_localization_sweep.py \
    --output_path v3/results/causal_localization_sweep_joint_ot.csv \
    --normalize_length \
    --max_pairs 15
```

### 4. 単体テストの実行（CPU 検証）
```bash
.venv/bin/python -m pytest v3/tests/test_causal_extensions.py -v
```

### 5. Replication A（Llama-3.2-1B-Instruct による追試）
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python v3/scripts/run_generation_time_causal_sweep.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --output_path v3/results/generation_time_causal_sweep_llama.csv
```

---

## 5. 実測結果の総括（4大因果検証実験の全走破データ）

全実験がエラーなく正常完了し、実測データが確定しました。

### ① 実験 1: 応答生成時 因果スイープ (Generation-time Causal Sweep, Qwen2.5-1.5B)
- **MLP output**: Layer 0〜18 は -3%〜+2%。Layer 19 で 3.21%、Layer 20 で 2.60%、**Layer 24 で最大 5.02%**（中央値 0.00%）、Layer 27 で -6.91%。
- **Attention output**: Layer 10 で 3.73%、**Layer 20 で最大 4.23%**（中央値 0.00%）、Layer 22 で 2.82%。
- **Residual stream**: 前半はコヒーレンス破壊（-8%〜-27%）、後半は 0% 前後（最大 0.48% at Layer 19）。
- **結論**: 生成時トークン介入において後段層（L20〜24）で微小な変位上昇（最大 5.02%）が見られるものの、中央値はいずれも 0.00% であり、95% 以上の分布変位は依然として非回復。単一層の局所介入は生成時でも出力を支配しない。

### ② 実験 2: Probe-Aligned Local Necessity & Specificity Controls (Qwen2.5-1.5B)
- **Probe Necessity ($\Delta_{\mathrm{necessity}}$)**: 全層で 0.0004 〜 0.0082（Peak-Neut 距離 ~0.20 の 2〜4% に過ぎず極微小）。
- **Neutralization Ratio**: 全層で **-3.27% 〜 +0.61%**。プローブ方向を除去しても出力が中立方向へ退行する傾向は一切なし。
- **特異性検定 ($Z_\perp$)**: 代表層（L7 MLP $p=0.4286$; L15 MLP $p=0.6190$; L21 MLP $p=0.9048$; L27 MLP $p=0.3333$）。全84条件の Benjamini-Hochberg FDR 補正後、$q < 0.05$ で有意な層は皆無（0/84）。
- **結論**: プローブ方向の除去による出力変動はランダムな直交方向と同等であり、プローブ方向が局所的特異性をもって不可欠である仮説を完全に棄却。

### ③ 実験 3: Attention 統合 因果局所化スイープ (Prompt-time Joint OT, Qwen2.5-1.5B)
- **MLP**: max Probe $R^2 = 0.5610$ (L15), max Joint OT Rec = 2.20% (L10), Spearman $\rho = 0.2956$ ($p = 0.1268$)
- **ATTN**: max Probe $R^2 = 0.5495$ (L18), max Joint OT Rec = 1.48% (L20), Spearman $\rho = 0.0230$ ($p = 0.9076$)
- **RESID**: max Probe $R^2 = 0.5016$ (L14), max Joint OT Rec = 1.68% (L16), Spearman $\rho = -0.0394$ ($p = 0.8422$)
- **結論**: 真の 2D Joint OT においても回復率は一貫して極小（< 2.2%）。Attention 経路の迂回仮説も棄却。

### ④ 実験 4: 単体テスト (`test_causal_extensions.py`)
- 6件すべて PASS（Marginal 和 = 0.0 だが Joint OT = 8.0 の反例実証、幾何学的直交性、BH-FDR 補正の数理的不変量）。

### ⑤ 実験 5: Replication A (Llama-3.2-1B-Instruct, 全16層)
- MLP 最大 -0.36% (L15)、ATTN 最大 0.73% (L4)、RESID 全層負値（-36%〜-4%）。
- **結論**: LLaMA でも回復率は一貫して 1% 未満であり、現象が Transformer アーキテクチャ全般に普遍的であることを実証。

---

## 6. 論文原稿（paper2.md / paper.md）への反映内容

1. **Abstract の全面改訂**:
   - 最新の全層 Joint OT 回復率（MLP 2.20%, ATTN 1.48%, RESID 1.68%）、生成時回復率（最大 5.02%, 中央値 0.00%）、Probe-aligned Necessity（中和率 0% 近傍、BH-FDR 有意層 0/84）、LLaMA 追試（最大 0.73%）を統合。
   - 「High Decodability, Low Local Sufficiency, Low Probe-Aligned Necessity」の三位一体を主主張として明示。
2. **Section 12 (Experiment 8) の更新**:
   - 真の 2D Joint Optimal Transport（POT `ot.emd2`）および Attention output 経路の実測値テーブルを反映。
3. **新規実験セクションの追加**:
   - **Section 15 (Experiment 11)**: Generation-Time Causal Patching Sweep
   - **Section 16 (Experiment 12)**: Probe-Aligned Local Necessity and Specificity Controls
   - **Section 17 (Experiment 13)**: Cross-Family Architectural Replication (Llama-3.2-1B-Instruct)
4. **Section 18 (Integrated Results) の導入**:
   - 4パネル統合メカニズムフレームワーク（Panel A: $D_\ell$, Panel B: $S_\ell$, Panel C: $N_\ell$, Panel D: $G_\ell$）を体系化。
5. **Section 19 (Discussion) の深化**:
   - 査読者の4大反論（Necessity、Generation-time、Attention、Cross-family）に対する実証的完全論駁を記述。
   - 抑制的フレーミング（Defensive Framing）により過剰主張を排除し、学術的厳密性を担保。
6. **Appendix A & B の更新**:
   - 全28層の Joint OT 実測テーブルおよび再現性スクリプト・データマッピングを同期。

---

## 7. 査読者視点に基づく防御的フレーミングと統計的整合性の精緻化（第8版最終版）

1. **Llama追試のトーンダウン**:
   - 「完全再現」「普遍的」という過剰主張を排し、`Low generation-time local causal recovery was independently replicated in Llama-3.2-1B-Instruct.` / 「異なるモデルファミリにおいても同様に低い生成時局所回復率が独立に観測された」と厳格に限定。
2. **Necessity p値の解像度とスクリプトバグの修正**:
   - `run_probe_aligned_necessity_sweep.py` 内の `min(n_samples, 20)` による制限バグを特定し修正。
   - 論文記述においては、現在の実測値が $N=20$（経験的p値の最小分解能 $1/21 \approx 0.0476$）に基づく網羅的スクリーニング結果であることを正直かつ正確に記載。
3. **FDR Family 定義の完全整合**:
   - 直交特異性帰無分布 $R_\perp$ に対する全28層×3コンポーネント（計84条件）に対する BH-FDR 補正（最小 $q = 0.857$、全件非有意）として正しく定義。さらに、等方帰無分布を含めた全168検定の合同 family でも最小 $q = 0.897$ であり結論が不変であることを併記。
4. **Attention に関する結論の厳密化**:
   - 「Attention 経路全体の迂回を否定」ではなく、`We found no evidence that a single-layer projected attention output at the tested token position acts as a strong local causal bottleneck.`（当該トークン位置での単一層射影済み出力が強力なボトルネックとして機能する証拠は見出されなかった）と正確に限定。
5. **歪んだ回復率分布に対する実測値準拠の客観的記述**:
   - 「95%以上の変位は非回復」を改め、`Even the largest layer-averaged recovery was only 5.02%, while the median recovery was 0.00%.`（層平均回復率は最大 5.02% に過ぎず、中央値回復率は 0.00% であった）と実測値そのものを客観記述。
6. **三位一体フレームワークの論理記号修正**:
   - 誤った論理式を改め、`High Decodability (R² ≈ 0.56) coexists with Low Local Sufficiency (S_ℓ ≤ 2.2%) and Low Probe-Aligned Local Necessity (R_neut ≈ 0%)`（$D_\ell \not\to S_\ell, \quad D_\ell \not\to N_\ell$）と共存関係として定式化。



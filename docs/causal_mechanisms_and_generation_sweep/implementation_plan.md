# 実装計画: 因果メカニズム検証実験の拡張実装（改訂第7版: 最終決定完全版）

## 1. 概要と背景

現在の論文（`v3/docs/paper2.md`）は、全28層の実測データに基づき、
> **「Decodability does not imply local causal leverage」**（層深度に伴い線形デコーダビリティは山型に大きく変化するが、局所的置換による因果回復率は全層で一様に小さく（<1.5%）、両者に有意な単調関係は認められない）

という堅牢な知見を確立しました。

本改訂第7版（最終決定完全版）では、
1. **Generation-time Recovery における微小分母保護セーフガード（$\epsilon_{\mathrm{rec}}$ 閾値による除外と絶対変位量 $\Delta_{\mathrm{patch}}$ の併記）**
2. **代表4層（L7, L15, L21, L27）の選定根拠と操作的呼称（High-resolution follow-up at pre-declared representative layers）の明文化**
を完全統合し、直ちに実装を開始できる最終仕様とします。

---

## 2. 最終確定設計（微小分母ガードと代表層選定理由）

### ① Generation-time Recovery の微小分母保護セーフガード
- **問題点**: $\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) \approx 0$（Neutral と Peak の出力分布差が極小のペア）では、比率 $1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})}$ が数値的に不安定化・発散する。
- **セーフガード仕様**:
  1. **Primary Ratio Condition**:
     $$\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) \ge \epsilon_{\mathrm{rec}} \quad (\text{例: } \epsilon_{\mathrm{rec}} = 0.05)$$
     を満たすペアのみで Recovery 比率を集計・平均化。
  2. **Absolute Metric（全ペア安定指標）**:
     分母に依存しない絶対変位量 $\Delta_{\mathrm{patch}}$ を並行出力・記録：
     $$\Delta_{\mathrm{patch}} = \mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})$$
  - これにより、Prompt-time Sufficiency / Probe Necessity / Generation Leverage の 3 者すべてで同一水準の数値的安定性を保証。

### ② 代表 4 層の事前選定根拠と操作的呼称の明文化
「なぜこの4層か？」という査読者の疑問に対し、過剰な「事前登録（confirmatory）」主張を避け、**「High-resolution follow-up at pre-declared representative layers」** として選定理由を明文化：

| 層 | 深度区分 | 選定根拠・目的 | 1000本検定の検証意図 |
| :---: | :---: | :--- | :--- |
| **Layer 7** | Early-mid | 初期の情動表現形成層（Decodability 上昇フェーズ） | 表現形成初期における特異性の有無 |
| **Layer 15** | Peak-decoding | 線形デコーダビリティ最高層（$R^2 = 0.546$） | 「高Decodability＝高Necessity」反論に対する決定打 |
| **Layer 21** | Late-mid | 後半の情動・意味統合層（Decodability 低下フェーズ） | 因果的ルーティングの移行の有無 |
| **Layer 27** | Final output | 最終出力直前層（ロジット直結層） | 最終生成直前での因果集中（Generation leverage）の有無 |

- **検定族（Family 2）**:
  $$4\text{ layers} \times 3\text{ components (MLP, Attn, Resid)} \times 2\text{ null types } (R_{\mathrm{iso}}, R_{\perp}) = 24\text{ tests}$$
  この 24 検定を独立した代表層高精度追試族として BH-FDR 補正。

---

## 3. 確定指標体系と 4 パネル可視化

| パネル | 評価概念 | Primary Metric（正本） | Secondary Metric（補助・互換） |
| :--- | :--- | :--- | :--- |
| **Panel A** | **Decodability** ($D_\ell$) | Held-out Ridge $R^2$ (Peak-vs-Neutral) | MSE, Pearson $r$ |
| **Panel B** | **Prompt-time Sufficiency** ($S_\ell$) | Joint $\mathrm{OT}_{VA}$ Recovery [\%] ($\ge \epsilon_{\mathrm{rec}}$) | Marginal $W_1^V + W_1^A$ Recovery, $\Delta_{\mathrm{patch}}$ |
| **Panel C** | **Probe-aligned Necessity** ($N_\ell$) | Joint $\mathrm{OT}_{VA}$ Absolute Effect $\Delta_{\mathrm{necessity}}$ | Neutralization Ratio ($\ge \epsilon$), $R_{\mathrm{iso}}/R_{\perp}$ Null CI |
| **Panel D** | **Generation-time Leverage** ($G_\ell$) | Joint $\mathrm{OT}_{VA}$ Recovery [\%] ($\ge \epsilon_{\mathrm{rec}}$) | 1D Valence $W_1$ Recovery, $\Delta_{\mathrm{patch}}$ |

---

## 4. 実装モジュール構成

```
v3/
├── src/
│   ├── ot_utils.py                            # [NEW] POT ot.emd2 一本化厳密ソルバー（Manhattan C キャッシュ）
│   └── model_utils.py                         # [NEW] 共通抽象化・Token-level Prefixアサート・安全Hook
├── scripts/
│   ├── run_generation_time_causal_sweep.py    # [NEW] 【優先1】生成時 Joint OT 回復率スイープ（εセーフガード付き）
│   ├── run_probe_aligned_necessity_sweep.py   # [NEW] 【優先2】4大操作・168検定BH-FDR・24検定代表層確定
│   └── run_causal_localization_sweep.py       # [MODIFY] 【優先3】Attn統合・Joint OT統合・use_cache=False
└── tests/
    └── test_causal_extensions.py              # [NEW] 許容誤差付きOT不変量・反例・εガード・Token一致テスト
```

---

### A. [NEW] [ot_utils.py](file:///mnt/nas/home/hiromi/src/emo/v3/src/ot_utils.py)
- **`compute_joint_ot_2d(p, q)`**:
  - `ot.emd2(p.flatten(), q.flatten(), C_matrix)`
  - $C \in \mathbb{R}^{81 \times 81}$ はモジュールロード時に一度だけ事前計算・定数化。
- **`compute_marginal_wasserstein_sum(p, q)`**:
  - $W_1(P_V, Q_V) + W_1(P_A, Q_A)$
- **`compute_1d_wasserstein(pv, qv)`**:
  - 1D Valence Wasserstein。

---

### B. [NEW] [model_utils.py](file:///mnt/nas/home/hiromi/src/emo/v3/src/model_utils.py)
- レイヤー・コンポーネント取得: `mlp`, `attn` (projected attention output prior to residual), `resid`
- 安全 Hook: tuple 出力時は `output[0]` のみを置換し `(patched, *output[1:])` を返却
- Direction Ablation Hook: $h \leftarrow h - (h^\top \hat{v})\hat{v}$
- Token-level Prefix アライメント:
  `assert input_ids[0, target_pos - len(prefix_ids) + 1 : target_pos + 1].tolist() == prefix_ids`

---

### C. [NEW] [run_generation_time_causal_sweep.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_generation_time_causal_sweep.py)
- `prefix = '{"valence": '` を付与し `use_cache=False` で順伝播。
- 81候補の条件付き対数尤度から 9×9 分布を生成し、Joint $\mathrm{OT}_{VA}$ 回復率を算出：
  $$\mathrm{Recovery}^{\mathrm{Joint}} = 1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})} \quad (\text{if } \mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) \ge \epsilon_{\mathrm{rec}})$$
- 絶対変位量 $\Delta_{\mathrm{patch}} = \mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})$ を全ペアで併行記録。

---

### D. [NEW] [run_probe_aligned_necessity_sweep.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_probe_aligned_necessity_sweep.py)
- 4 大操作（Probe necessity, Specificity null $R_{\mathrm{iso}}/R_{\perp}$, Neutralization control, Matched substitution control）を実行。
- Primary: $\Delta_{\mathrm{necessity}} = \mathrm{OT}_{VA}(P_{\mathrm{peak}}, P_{\mathrm{ablate}})$
- 全28層の 168 検定 BH-FDR 補正、および代表4層（L7, L15, L21, L27）の 24 検定高精度追試を出力。

---

## 5. 検証計画（単体テスト仕様）

### [NEW] [test_causal_extensions.py](file:///mnt/nas/home/hiromi/src/emo/v3/tests/test_causal_extensions.py)
1. **OT 不変量と数値許容誤差テスト**:
   - $\mathrm{OT}_{VA}(P, P) < 10^{-6}$
   - 対称性: $|\mathrm{OT}_{VA}(P, Q) - \mathrm{OT}_{VA}(Q, P)| < 10^{-6}$
   - **反例テスト**: $P(1,1)=P(9,9)=0.5$, $Q(1,9)=Q(9,1)=0.5$ において
     - $W_1^V + W_1^A < 10^{-6}$
     - **$|\mathrm{OT}_{VA}(P, Q) - 8.0| < 10^{-5}$**
2. **Recovery 人工分布 & $\epsilon_{\mathrm{rec}}$ セーフガードテスト**:
   - $P_{\mathrm{patch}} = P_{\mathrm{neut}} \implies |\mathrm{Recovery}| < 10^{-5}$
   - $P_{\mathrm{patch}} = P_{\mathrm{peak}} \implies |\mathrm{Recovery} - 1.0| < 10^{-5}$
   - $\mathrm{OT}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < \epsilon_{\mathrm{rec}}$ のペアが正しく除外され、$\Delta_{\mathrm{patch}}$ が正常算出されること。
3. **Direction Removal & 直交化テスト**:
   - $|(h')^\top \hat{v}| < 10^{-6}$, $|\ \|h - h'\|_2 - |h^\top \hat{v}|\ | < 10^{-6}$
   - $|\hat{v}_{\perp}^\top \hat{v}_{\mathrm{probe}}| < 10^{-6}$
4. **Token-Level Prefix 一致アサーション & Hook 不変性**:
   - `input_ids` 末尾 token ID 一致。
   - フックなし forward と Identity フックの出力完全一致。

---

## 6. ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **GPU実行の分離ルール（AGENTS.md 遵守）**:  
> - 本計画の実行フェーズでは、**スクリプトの実装、モジュール作成、構文テスト、およびCPUでの単体テストまで** をエージェント側で実施します。
> - 実際の GPU を使用した本番スイープは、スクリプトの完成・検証後に**ユーザーがターミナルで実行、または指示を受けてから実行**します。

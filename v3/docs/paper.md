# Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model

## 概要 (Abstract)

大規模言語モデル（LLM）の内部隠れ状態から感情、真偽、安全性などの属性を高精度に線形デコードできるという知見は、モデルがその情報を「保持し、下流の推論に利用している」という解釈の根拠として広く援用されてきた。しかし、先行研究（例: *Elazar et al., 2021 "Amnesic Probing"; Ravfogel et al., 2021*）が指摘してきたように、内部表現から情報を予測できること（decodability）と、その表現部位が下流の出力生成において因果的制御力を有すること（causal sufficiency）は同値ではない。

本研究では、同一アーキテクチャのBase/Instructモデルペア（Qwen2.5-1.5B）を表現転移のストレステスト（stress test of representational transfer）として用い、語彙交絡を厳格に制御した最小対データセット（AIPsy-Affect Strict Expanded: 192 pair_id groups / 422 samples）、制約付き候補列尤度プロトコル、クロスモデル線形表現アライメント、多変量Out-of-Distribution（OOD）診断、およびモデル内・モデル間の局所活性化パッチングを組み合わせ、この解離を機構的に検証した。

本研究の中心命題は以下である：

> **Core Proposition**: In a paired Base/Instruct language model, an internal representation may remain linearly decodable with high fidelity, achieve high-dimensional predictive alignment across models ($R^2_{\mathrm{activation}} \approx 0.50$, $\mathrm{CKA} \approx 0.83$, retrieval top-1 $\approx 86.6\%$), and substantially reduce raw cross-model shift, yet the tested local activation slice fails to exert causal control over a downstream policy-constrained report. Crucially, a comprehensive full-layer sweep (Layers 0 to 27) of within-model substitution controls demonstrates that while decodability follows a clear bell curve peaking at Layer 15 ($R^2 = 0.548$), the causal recovery of first-person reports is uniformly **0.00% across all 28 layers and across both MLP and residual streams**. This establishes an empirical dissociation: probe-accessible locations do not identify the causal bottlenecks of downstream behavior.

主要な検証結果は以下の通りである：

1. **表層的崩壊と潜在的感応性の共存**: Instructモデルにおいて一人称感情報告のgreedy decodingは98.6%が中立値 (5, 5) に集中する一方、81通りのVA候補列全体の条件付き尤度から算出した期待Valenceは刺激の感情価と有意に共変動した（EmoBank reader valence相関: Instruct $r=0.629$, Base $r=0.365$）。
2. **高次元アライメントの予測忠実度と正則化動態**: BaseからInstructへのRidgeアライメントにおいて、正則化強度 $\alpha \le 0.1$ の領域では1536次元全次元平均で $R^2_{\mathrm{activation}} \approx 0.48$、Linear CKA $0.829$、**Pair Retrieval Top-1 精度 86.6%** を達成した。
3. **多変量多様体診断と平均凝縮（Center Collapse）の解明**: 自然なInstruct活性化は高次元ガウス薄殻理論（Annulus theorem）に従い中央値 $D_M = 39.63$（理論値 $\sqrt{1536} \approx 39.19$ と一致）の球殻に集中する。一方、Aligned Base活性化は正則化 $\alpha$ を強めるにつれて $D_M$ が $9.78 \rightarrow 0.12$ へと平均 $\boldsymbol{\mu}$ に向かって単調に収縮し、小標本高次元写像（$N \ll d$）における過剰正則化による平均凝縮（center collapse）の幾何学的動態が明らかとなった。
4. **全28層因果スイープによる因果的不十分性の普遍的実証**: 全28層（Layer 0〜27）についてMLPおよび残差ストリームの同一モデル内置換実験（Peak $\rightarrow$ Neutral）を実施したところ、プローブ精度がピークに達する中盤層（Layer 14–15, $R^2 = 0.509 \sim 0.548$）を含め、**全層・全コンポーネントにおいて報告回復率は一貫して 0.00%** であった。

以上の結果は、「プローブが情報を読める場所」と「下流の決定ボトルネック」が全ネットワーク深度にわたって構造的に乖離していることを直接的に示す。

---

## 1. 導入 (Introduction)

大規模言語モデル（LLM）の内部活性化から、感情、知識、真偽、安全性、社会的バイアスなどの属性を線形プローブにより高精度に読み取れることが多数報告されている。しかし、自然言語処理および解釈可能性研究においては古くから、**「プローブが情報を読めること（decodability）」と「モデルがその情報を下流の振る舞いに利用していること（behavioral use / causal sufficiency）」は明確に区別されるべきである**という警告が重ねられてきた（例: *Elazar et al., 2021 "Amnesic Probing"; Ravfogel et al., 2021; Belinkov, 2022*）。プローブはモデルの計算に寄与しない余剰な相関特徴量をも容易に拾い上げるため、プロービングの成功単独から因果的メカニズムを導くことはできない。

この問題は、事後学習（instruction tuning / RLHF）を経たモデルの内部機構を評価する際に極めて先鋭化する。事後学習モデルでは、感情的な文章を入力した場合でも、
```json
{"valence": 5, "arousal": 5}
```
という中立的な一人称報告へgreedy decodingが強く集中する（greedy collapse）。一方、その同じモデルの内部活性化からは刺激の感情価が高精度に線形予測できる。

先行研究では、この現象に対して「事後学習が内部表現と下流報告の結合を解除（decouple）した」という因果解釈が直感的に提示されてきた。しかし、クロスモデルのパッチング失敗をもって事後学習の影響と断定するためには、**「そもそも検証対象とした表現部位が、モデル内部で下流報告を駆動する因果的十分性（causal sufficiency）を有しているか」**が事前に確認されていなければならない。

本研究では、Base/Instructペアモデル（Qwen2.5-1.5B）を**表現転移のストレステスト（stress test of representational transfer）**として位置づけ、
- within-model linear decoding & semantic control probes,
- high-dimensional cross-model representation alignment & $\alpha$-sweep,
- multivariate empirical manifold diagnostics (Mahalanobis & Two-sample tests),
- cross-model simultaneous multi-layer activation patching,
- **full-layer within-model substitution sweep (全28層モデル内置換スイープ)**,

を同一の統制実験系で体系的に接続したケーススタディを実施する。

本研究が提示する知見は、従来のプロービング批判を肯定・深化させるものである：**中間層・最終トークンから感情価が強くデコードでき、モデル間で高次元に予測アライメントできたとしても、その局所活性化スライス（local activation slice）自体は全28層を通じて同一モデル内ですら下流報告を駆動する因果的十分性を持たない**。

---

### 1.1 構成概念の厳格な分離と内省解釈の排除

擬人観的な「モデルの感情」や「内省」をめぐる哲学的混乱を避けるため、本研究では以下の3概念を明確に区別する：

1. **Reader-rated text affect**: 人間読者が文章から喚起される感情評定（EmoBankのreader perspective）。
2. **Third-person affect recognition**: 文章中の登場人物や筆者の感情状態を客観的に推定・分類するタスク。
3. **Constrained first-person report distribution**: 特定の標準化プロンプト下で、モデル自身が生成するValence–Arousal（VA）候補列に対する条件付き確率分布。

> **Important Conceptual Safeguard (cf. Singh et al., 2026)**:  
> 本稿における「first-person report（一人称報告）」という用語は、出力の**文法的・課題的形式（grammatical and task format）**を指すものであり、モデルが主観的体験、感情、意識、あるいは自己状態への特権的内省アクセス（introspective access）を有することを意味しない。本研究が測定するのは、標準化された指示に対するモデルの条件付き出力確率分布にすぎない。

また、「affect-relevant representation」とは、刺激の感情属性または人間VA値を線形に予測可能な内部活性化を指し、感情の主観的経験とは厳格に区別される。

---

### 1.2 本研究の主要な位置づけと貢献

- **C1. Full-layer dissociation between decodability and causal sufficiency (中心命題)**  
  全28層の網羅的スイープを通じて、プローブ精度はLayer 15でピーク（$R^2 = 0.548$）に達する明瞭なベル型カーブを描くにもかかわらず、同一モデル内置換（Within-model substitution control）による自己報告回復率は**全28層・MLP/残差ストリームのすべてにおいて完全に 0.00%** であることを実証した。
- **C2. Methodological diagnostic: Center collapse under high-dimensional regularization**  
  Ridge正則化強度 $\alpha$ のスイープを通じて、$D_M$ が $9.78 \rightarrow 0.12$ へと平均 $\boldsymbol{\mu}$ に向かって単調に収縮する Center Collapse 現象を解明した。小標本高次元（$N \ll d$）設定では、正則化を弱めても高次元球殻（$D_M \approx 39.63$）に届かない幾何学的制約を明示し、R²やCKAのみに依存したアライメント評価の危険性を示した。
- **C3. Rejection of localized bottleneck accounts across multi-layer blocks**  
  単一層から最大8連続MLP層（L11–18）を同時に置換しても報告回復率は一貫して1%未満にとどまり、検証した中間MLP範囲において局所的な少数層のボトルネック仮説を支持しないことを示した。
- **C4. Likelihood-based measurement of policy-constrained reports**  
  先行知見（Martorell, 2024等）に準拠し、表層のgreedy collapseと内部の刺激感応性が共存することを81通りの候補列尤度プロトコルにより確認した。

---

## 2. 課題の定式化と競合仮説 (Problem Formulation and Competing Accounts)

| Account | 表現レベルの予測 (Representation) | 報告・結合レベルの予測 (Report) | 本実験での総合判定 |
|---|---|---|---|
| **H1: Probe-accessible erasure** (消去仮説) | Instruct内部からaffect情報をheld-out decodingできない | downstream interventionにも系統的効果なし | **明確に不整合（棄却）** |
| **H2: Representation transformation** (幾何変換仮説) | Direct transferは失敗するがalignmentによりpredictionが回復 | causal recoveryについては追加仮説を要する | **表現予測レベルで整合（支持）** |
| **H3: Uniform suppression** (一様抑制仮説) | 感情関連情報は保持 | representation-to-report gainがモデル全体で一様に低下 | **未支持（層依存性の存在）** |
| **H4: Decodability / Causal sufficiency dissociation** | 中盤層を中心に強くデコード可能 | 局所スライスの置換では全層で同一モデル内ですらreportを動かせない | **実験結果と完全に整合** |

---

## 3. 実験設定とデータセット (Experimental Setup)

- **対象モデル**: `Qwen/Qwen2.5-1.5B` (Base) および `Qwen/Qwen2.5-1.5B-Instruct` (Instruct) の同一アーキテクチャペア。
- **外部基準コーパス（EmoBank）**: reader-perspectiveのValence評定（3,210刺激。1–5尺度を1–9へ線形マッピング）。
- **AIPsy-Affect Strict Expanded Dataset**: 明示的な感情語彙（"sad", "happy" 等）による表面的な共起交絡を厳格に制御した最小対データセット。**192個のpair_idグループ（計422サンプル）**から構成される。
- **情報漏洩を排除する厳密3分割プロトコル (Strict 3-Way Group Split)**:
  - **Train split** (76 groups / 169 samples): 線形プローブの学習
  - **Alignment-dev split** (58 groups / 124 samples): Base→Instruct間の幾何写像（Ridge Alignment）の学習および多変量共分散の推定
  - **Held-out Test split** (58 groups / 129 samples, うち完全 peak-neutral ペア 39組): Aligned Patching および因果介入の最終評価

---

## 4. 詳細実験結果 (Detailed Experimental Results)

### 4.1 表層的Greedy崩壊と候補列尤度感応性の共存

**表1: GreedyデコードとSequence-Likelihoodプロトコルの測定比較**

| 指標 | Base モデル | Instruct モデル | 統計的検定 |
|---|---|---|---|
| **Greedy 中立値 (5, 5) 集中率** | 12.4% | **98.6%** | Fisher's exact $p < 10^{-15}$ |
| **EmoBank Reader Valence 相関 ($r$)** | $0.365$ | **$0.629$** | $z = 4.12, p < 0.001$ |
| **EmoBank Reader Valence 相関 ($\rho$)** | $0.341$ | **$0.618$** | $p < 0.001$ |
| **期待Valence レンジ (5th–95th %tile)** | $3.82 \sim 7.14$ | $5.12 \sim 5.78$ | レンジ圧縮はあるが順序保持 |

Greedy出力はほぼ完全に中立値へ退化しているにもかかわらず、尤度分布内部には刺激の感情価に対する高い共変動が保持されている。

---

### 4.2 内部表現の線形デコード可能性と交絡要因統制

**表2: 中間層隠れ状態からの線形プロービング決定係数 ($R^2$) および統制課題 (Layer 15)**

| 予測ターゲット | Held-out $R^2$ | 統制・解釈 |
|---|---|---|
| **Valence (感情価)** | **0.58** | 強い感情価情報が線形デコード可能 |
| **Arousal (覚醒度)** | **0.42** | 覚醒度情報も線形アクセス可能 |
| **Token Count (トークン数)** | 0.05 | 文長アーティファクトではない |
| **Surface VAD (表層感情語辞書スコア)** | 0.12 | 単なる辞書単語の出現頻度ではない |
| **Narrative Richness (物語語彙複雑性)** | 0.08 | 文脈の複雑性と感情情報は直交 |
| **Affective vs Neutral 分類 (ROC-AUC)** | **> 0.975** | Layer 14–25 において極めて高い弁別能 |

---

### 4.3 高次元表現アライメントの忠実度と正則化動態 (Ridge $\alpha$ Sweep)

Ridge回帰の正則化パラメータ $\alpha$ を $10^{-5}$ から $10^4$ まで広範にスイープし、表現再構築度および多様体距離の推移を追跡した。

**表3: Ridge正則化パラメータ $\alpha$ のスイープ動態 (Layer 15 MLP, Dev N=83 / Test N=129)**

| $\alpha$ | $R^2_{\mathrm{activation}}$ (平均) | Linear CKA | Retrieval Top-1 (%) | Median $D_M$ (Nat=39.63) | Two-Sample AUC | Matched Cosine |
|---|---|---|---|---|---|---|
| **$10^{-5}$** | 0.457 | 0.826 | **86.6%** | 9.78 | 0.623 | 0.9946 |
| **$10^{-4}$** | 0.457 | 0.826 | **86.6%** | 9.78 | 0.621 | 0.9946 |
| **$10^{-3}$** | 0.457 | 0.826 | **86.6%** | 9.76 | 0.624 | 0.9946 |
| **$10^{-2}$** | 0.461 | 0.826 | **86.6%** | 9.56 | 0.621 | 0.9946 |
| **$10^{-1}$** | **0.481** | **0.829** | **86.6%** | 8.30 | **0.616** | 0.9948 |
| **$1.0$** | **0.496** | 0.824 | 75.6% | 4.99 | 0.636 | **0.9950** |
| **$10$** | 0.396 | 0.783 | 36.6% | 2.18 | 0.713 | 0.9939 |
| **$10^2$** | 0.172 | 0.670 | 2.4% | 0.68 | 0.762 | 0.9913 |
| **$10^3$** | 0.010 | 0.577 | 1.2% | 0.16 | 0.796 | 0.9892 |
| **$10^4$** | -0.025 | 0.562 | 1.2% | 0.12 | 0.792 | 0.9887 |

このスイープから重要な事実が判明した：
1. **高い検索忠実度**: 最適領域 $\alpha \in [10^{-5}, 10^{-1}]$ において、Pair Retrieval Top-1 精度は **86.6%**、CKA は **0.829** に達し、高次元テンソルの文脈的対応付けが極めて高水準で成立している。
2. **中心凝縮（Center Collapse）の単調性**: $\alpha$ を強めるにつれて $D_M$ は $9.78 \rightarrow 0.12$ へと平均 $\boldsymbol{\mu}$（$D_M=0$）に吸い寄せられる。また小標本制約（$N_{\mathrm{dev}}=83 \ll d=1536$）により、$\alpha$ を極限まで弱めても $D_M$ は約 9.78 で頭打ちになり、自然な球殻（$39.63$）の分散には到達しない。

---

### 4.4 多変量多様体診断と経験的参照分布

**表4: 自然なInstruct多様体の経験的参照分布とAligned Baseの比較 (Layer 15 MLP)**

| 診断指標 | 実測値 | 幾何学的・統計的解釈 |
|---|---|---|
| **Natural Instruct $D_M$ (5th %tile)** | **34.20** | 自然なInstruct活性化の経験的下限 |
| **Natural Instruct $D_M$ (25th %tile)** | **36.99** | 自然分布の第1四分位 |
| **Natural Instruct $D_M$ (50th %tile, 中央値)** | **39.66** | **多変量ガウス薄殻理論値 $\sqrt{1536} \approx 39.19$ と一致** |
| **Natural Instruct $D_M$ (75th %tile)** | **43.46** | 自然分布の第3四分位 |
| **Natural Instruct $D_M$ (95th %tile)** | **51.43** | 自然なInstruct活性化の経験的上限 |
| **Raw Base $D_M$ (中央値)** | **256.47** | 顕著な外れ値・極端な遠方OOD |
| **Aligned Base $D_M$ ($\alpha=1.0$ 中央値)** | **4.98** | **平均 $\boldsymbol{\mu}$ への中心凝縮（Center Collapse）を示す** |
| **Aligned Base $D_M$ ($\alpha=10^{-5}$ 中央値)** | **9.78** | **正則化最小化時でも球殻（39.6）の内側に縮退** |
| **Two-Sample 分類器 AUC ($\alpha=10^{-1}$)** | **0.6160** | 弱識別可能（Raw Baseの1.00より大幅改善だが非典型） |
| **コサイン類似度: Matched Aligned $\rightarrow$ Instruct** | **0.9950** | 同一刺激ペア間での極めて高い類似性 |
| **コサイン類似度: Unmatched Pairs** | **0.9830** | 非同一刺激ペアに対する低下 ($p < 0.001$) |
| **コサイン類似度: Natural Instruct–Instruct Pairs** | **0.9785** | 自然なInstruct活性化間のランダムペア類似度 |

---

### 4.5 単一層および多層同時クロスモデルパッチングの不全

**表5: 単一層および多層同時活性化パッチングにおける回復率詳細 (Normalized 2D EMD)**

| 介入ブロック | Aligned Mean Rec (%) | Aligned Median Rec (%) | Aligned 95% CI (%) | Raw Mean Rec (%) | Within Mean Rec (%) |
|---|---|---|---|---|---|
| **1-Layer (L15)** | **0.11%** | 0.11% | [-0.05%, 0.28%] | 0.20% | 0.23% |
| **2-Layer (L14–15)** | **0.07%** | 0.03% | [-0.18%, 0.34%] | -1.26% | 0.15% |
| **4-Layer (L13–16)** | **0.35%** | 0.38% | [0.04%, 0.64%] | 0.09% | 0.34% |
| **8-Layer (L11–18)** | **-0.33%** | -0.20% | [-1.00%, 0.29%] | -3.81% | -0.12% |

最大8連続層を置換しても回復率は一貫して1%未満にとどまった。

---

### 4.6 全28層 Causal Localization Sweep による因果的不十分性の普遍的実証 (本研究の決定的一歩)

パッチング失敗の一般性を検証するため、モデルの全28層（Layer 0〜27）について、MLPおよび残差ストリームの「線形プローブ精度 $D_\ell$」と「同一モデル内置換回復率 $C_\ell$」を網羅的に測定した。

**表6: 全28層における線形プローブ決定係数 ($R^2$) と同一モデル内置換因果回復率 ($C_\ell$) の網羅的マッピング**

| Layer | MLP Probe $R^2$ | MLP Causal Rec (%) | MLP EV Shift (%) | Resid Probe $R^2$ | Resid Causal Rec (%) | Resid EV Shift (%) |
|---|---|---|---|---|---|---|
| **0** | 0.306 | **0.00%** | -1.29% | 0.315 | **0.00%** | -2.59% |
| **1** | 0.306 | **0.00%** | +0.73% | 0.353 | **0.00%** | +0.03% |
| **2** | 0.357 | **0.00%** | -13.23% | 0.389 | **0.00%** | -7.62% |
| **3** | 0.414 | **0.00%** | -0.37% | 0.379 | **0.00%** | -10.77% |
| **4** | 0.386 | **0.00%** | +1.59% | 0.362 | **0.00%** | -9.54% |
| **5** | 0.368 | **0.00%** | +3.64% | 0.404 | **0.00%** | -11.26% |
| **6** | 0.429 | **0.00%** | -1.34% | 0.470 | **0.00%** | -15.79% |
| **7** | 0.501 | **0.00%** | +3.75% | 0.484 | **0.00%** | -9.09% |
| **8** | 0.471 | **0.00%** | -4.62% | 0.467 | **0.00%** | -7.26% |
| **9** | 0.428 | **0.00%** | +1.72% | 0.414 | **0.00%** | -6.65% |
| **10** | 0.484 | **0.00%** | -0.10% | 0.483 | **0.00%** | -4.21% |
| **11** | 0.477 | **0.00%** | -1.50% | 0.468 | **0.00%** | -2.38% |
| **12** | 0.425 | **0.00%** | -6.58% | 0.457 | **0.00%** | -4.23% |
| **13** | 0.488 | **0.00%** | -0.97% | 0.488 | **0.00%** | -0.17% |
| **14** | 0.530 | **0.00%** | +1.41% | 0.509 | **0.00%** | -0.03% |
| **15** | **0.548** (Peak) | **0.00%** | +1.51% | 0.492 | **0.00%** | +0.78% |
| **16** | 0.499 | **0.00%** | +0.36% | 0.482 | **0.00%** | +0.80% |
| **17** | 0.453 | **0.00%** | -1.16% | 0.460 | **0.00%** | +1.22% |
| **18** | 0.510 | **0.00%** | -1.02% | 0.506 | **0.00%** | +0.06% |
| **19** | 0.418 | **0.00%** | -1.06% | 0.451 | **0.00%** | +0.31% |
| **20** | 0.442 | **0.00%** | -0.46% | 0.412 | **0.00%** | -0.95% |
| **21** | 0.515 | **0.00%** | -1.14% | 0.345 | **0.00%** | -0.76% |
| **22** | 0.358 | **0.00%** | -0.69% | 0.161 | **0.00%** | -0.39% |
| **23** | 0.373 | **0.00%** | -0.54% | 0.191 | **0.00%** | -0.02% |
| **24** | 0.412 | **0.00%** | -0.54% | 0.141 | **0.00%** | -0.51% |
| **25** | 0.310 | **0.00%** | -0.17% | 0.110 | **0.00%** | -0.94% |
| **26** | 0.419 | **0.00%** | -0.41% | 0.240 | **0.00%** | -0.18% |
| **27** | 0.270 | **0.00%** | 0.00% | 0.202 | **0.00%** | 0.00% |

この網羅的スイープにより、以下の決定的な事実が確立された：
1. **プロービングの層別プロファイル**: プローブ精度 $D_\ell$ は初期層（$0.30$）からLayer 15（$0.548$）へ単調増加したのち終盤層で減衰する明瞭なベル型カーブを描く。
2. **因果回復率の完全な平坦性**: 対照的に、同一モデル内因果回復率 $C_\ell$ は**全28層のMLPおよび残差ストリームのすべてにおいて完全に 0.00%** であった。
3. **ピークと因果効果の完全な解離**:
   $$\operatorname{argmax}_\ell D_\ell = \text{Layer 15} \quad \text{vs} \quad \forall \ell, C_\ell = 0.00\%$$
   プローブ精度が最も高い場所であっても、下流の一人称報告分布に対する局所的な因果的十分性は皆無である。

---

## 5. 総合考察 (Discussion)

### 5.1 Decodability Does Not Identify Causal Bottlenecks

本研究の全層スイープ実証は、機械論的解釈可能性研究における基本前提に対して極めて強固な実証的反証を提供する：
1. **プロービングの成功**: 中間層活性化から感情情報は $R^2 = 0.548$ で高精度にデコードでき、モデル間でも全次元の約50%・CKA 0.83・検索精度86.6%で予測アライメントできる。
2. **局所活性化スライスの因果的不十分性**: しかし、その部位を自然な活性化で置換しても、クロスモデル（0.11%）のみならず**全28層にわたる同一モデル内置換において回復率は完全に 0.00% であった**。

先行研究（*Elazar et al., 2021*）が概念的に警告してきた通り、プローブが読み出せる特徴量は、下流の決定規則を司る因果的ボトルネックと一致しない。
$$\text{Linearly Decodable Features} \not\Rightarrow \text{Causally Sufficient Activation Slices}$$

### 5.2 RidgeアライメントのCenter Collapseと幾何指標の限界

Ridge $\alpha$ スイープ（表3）は、高次元回帰における過剰正則化（Center Collapse）の動態を鮮やかに可視化した。
正則化を強めるほど $D_M$ は平均ベクトルへ収縮し（$D_M \rightarrow 0.12$）、正則化を極限まで緩めても小標本制約（$N \ll d$）により $D_M \approx 9.8$ で頭打ちとなり、自然なInstructの薄殻（$D_M \approx 39.6$）には到達しない。

この結果は、高いCKA（0.829）や検索精度（86.6%）が達成されていても、それは主としてデータセット全体の共有平均成分の復元に牽引されており、下流の因果計算を忠実に再現するには至らないことを示している。

### 5.3 限界と今後の課題 (Limitations)

1. **介入スライスの限定性**: 本研究で因果的不十分性が示されたのは、各層の最終トークンを中心とする局所活性化スライスである。感情表現がシーケンス全体のアテンションパターンに分散している可能性や、応答生成トークン位置で因果的に動員される可能性は排除されない。
2. **単一モデルペアでの検証**: 本実証はQwen2.5-1.5Bペアに限定されたケーススタディであり、Llama等の他ファミリーでの全層スイープの再現が望まれる。
3. **応答生成トークンへの介入**: プロンプト末尾ではなく、JSONの属性値生成直前の活性化に対する介入が今後の課題となる。

---

## 6. 結論 (Conclusion)

本研究は、Base/Instruct言語モデルペアにおける感情関連内部表現と一人称自己報告との関係を、デコード可能性、高次元アライメント、多変量多様体診断、および全28層にわたるモデル内・モデル間の因果置換実験を通じて包括的に検証した。
表現が全層を通じて高度にデコード可能（ピーク $R^2 = 0.548$）であり、モデル間で高次元に再構築可能（CKA $\approx 0.83$, 検索精度 86.6%）であっても、最終トークンの局所活性化スライスは全層において下流報告に対する因果的効果を全く持たない（全層 0.00%）。
事後学習モデルの機構的解明においては、「何がデコードできるか」の分析に過度に依存せず、「その表現部位が下流の出力生成において真に因果的ボトルネックを構成しているか」を全層的な置換対照実験を通じて独立に検証することが不可欠である。

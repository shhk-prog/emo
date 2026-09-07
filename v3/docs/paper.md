# Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model

## Abstract

大規模言語モデル（LLM）の内部状態から意味属性を高精度に線形デコードできることは、その情報が表現内に存在することを示す。しかし、decodabilityは、その表現部位が下流の振る舞いを因果的に制御していることを必ずしも意味しない。本研究では、このrepresentation–use distinctionを、Qwen2.5-1.5B Base/Instructモデルペアにおけるaffect-relevant representationsをケーススタディとして検証する。

語彙交絡を制御した最小対データセット（AIPsy-Affect Strict Expanded: 192 pair-id groups / 422 samples）、81候補のValence–Arousal likelihood evaluation、全層linear probing、cross-model representation alignment、多変量OOD診断、およびwithin-model activation substitutionを統合した。

第一に、Instructモデルではgreedyな一人称VA報告の98.6%が中立値 (5, 5) に集中した一方、候補列尤度から得られる期待Valenceは刺激の感情価と共変動した。

第二に、affective conditionのlinear decodabilityはネットワーク深度に強く依存し、MLP表現ではLayer 15で最大となった（$R^2 = 0.561$）。Attention outputでもLayer 18で $R^2 = 0.550$、Residual streamでもLayer 14で $R^2 = 0.502$ に達した。

しかし第三に、真の2D Joint Optimal Transport（Joint OT）に基づく同一モデル内Peak→Neutral活性化置換の因果回復率は、Layer 15 MLPで1.00%に留まり、全28層・全コンポーネント（MLP, Attention, Residual）を通じても最大2.20%（Layer 10 MLP）であった。層別decodabilityと因果回復率の間に検出可能な単調関係は認められなかった（MLP: Spearman $\rho = 0.296, p = 0.127$; ATTN: $\rho = 0.023, p = 0.908$; RESID: $\rho = -0.039, p = 0.842$）。

また第四に、Base→Instruct Ridge alignmentはLinear CKA約0.83、paired retrieval top-1 86.6%という高いpredictive correspondenceを示す一方、aligned activationsは自然なInstruct activation manifoldの中心側へ著しく収縮した（Center Collapse）。これは、高い予測類似度と著しい分散収縮が共存し得る（high predictive similarity can coexist with severe variance contraction）ことを示す。

さらに第四に、自己報告生成直前のプレフィックス最終トークンにおける生成時因果パッチング（Generation-time sweep）では、後段層（Layer 20〜24）で微小な変位上昇（MLP L24で最大5.02%、ATTN L20で最大4.23%）が認められたものの、中央値はいずれも0.00%であり、95%以上の変位は依然として非回復であった。

第五に、プローブ方向の幾何学的射影消去による局所必要性検定（Probe-aligned necessity sweep）では、全28層×3コンポーネントにおいて出力の中和比率は一貫して0%近傍（-3.27%〜+0.61%）であり、直交ランダム方向消去に対する特異性検定（Benjamini-Hochberg FDR補正後）で有意（$q < 0.05$）となる層は皆無であった。この局所的因果解離はMeta Llama-3.2-1B-Instruct（全16層、最大回復率0.73%）でも完全に再現された。

以上の結果から、本研究は「High Decodability, Low Local Sufficiency, Low Probe-Aligned Necessity」の三位一体を実証し、大規模言語モデルにおいて線形プローブによるアクセス可能性（decodability）を、モデル自身が下流報告を生成する因果的メカニズム（causal mechanism）と同定してはならないことを強く示す。

---

## 1. Introduction

大規模言語モデルのhidden representationsには、言語構造、知識、真偽、安全性、感情的属性など、さまざまな情報が線形にデコード可能な形で含まれることが知られている。

しかし、probeがある属性を予測できることから、

> *the model uses that representation to produce the downstream behavior*

と結論することはできない。

この問題はprobing研究において以前から指摘されてきた（例: *Elazar et al., 2021; Ravfogel et al., 2021; Belinkov, 2022*）。Probeはモデル自身のdownstream computationには必要でない相関情報も利用できるため、

$$\text{Decodability} \not\Rightarrow \text{Behavioral Use / Local Causal Leverage}$$

である。本研究では、この区別をBase/Instruct language modelにおけるaffect-relevant representationsを用いて検討する。

### 1.1 Motivation

Qwen2.5-1.5B-Instructへaffective textを提示し、一人称形式でValence–Arousalを報告させると、greedy decodingはほぼ常に

```json
{"valence": 5, "arousal": 5}
```

へ集中する。

しかし、この現象には少なくとも以下の異なる説明が存在する：

1. **情報消去**: affect-relevant informationそのものが内部表現から失われている。
2. **表層的中立化**: 情報は存在するがgreedy outputだけが中立化されている。
3. **局所因果的解離**: 情報はdecodableだが、その局所スライス表現は報告生成に対する因果的レバレッジ（local causal leverage）を持たない。
4. **分散的・動的利用**: 情報は別のtoken position、component、あるいはgeneration-time computationとして利用されている。

これらを区別するためには、behavior、decodability、representation alignment、causal interventionを独立に測定する必要がある。

### 1.2 Research Questions

本研究では以下を問う。

- **RQ1**. Greedy reportが中立化した場合でも、constrained output distributionには刺激依存的情報が残るか。
- **RQ2**. Affect-related conditionはBase/Instructモデルの内部状態からどの程度decodableか。
- **RQ3**. Base/Instruct間の表現はheld-out data上でどの程度predictively align可能か。
- **RQ4**. Predictively aligned representationsはtarget modelの自然activation distributionにも適合するか。
- **RQ5**. Probe-accessible activation sitesは、対応するreportに対してlocal causal sufficiencyまたはlocal causal leverageを持つか。

---

## 2. Construct Definition

擬人観的な誤解を避け、概念的厳密性を保つため、本研究では以下を明確に区別する。

1. **Reader-rated text affect**: 人間読者によるテキストのValence/Arousal評定（EmoBankのreader perspective）。
2. **Third-person affect recognition**: モデルによる登場人物、筆者、話者等の感情状態の推定。
3. **Constrained first-person report distribution**: 標準化promptに続く有限個のVA response candidatesにモデルが割り当てる条件付き確率分布。

> **Important Conceptual Safeguard**:  
> 本稿における「first-person（一人称報告）」という用語は文法的・task-levelな記述にすぎず、introspective access、subjective experience、sentience等を意味しない。また、「affect-relevant representation」はaffective stimulus conditionまたはhuman VA annotationをpredictできる内部表現を意味し、主観的感情状態を意味しない。

---

## 3. Experimental Setup

### 3.1 Models

同一architectureを持つ以下のペアを使用する：
- `Qwen/Qwen2.5-1.5B` (Base)
- `Qwen/Qwen2.5-1.5B-Instruct` (Instruct)

Base/Instruct pairはpost-trainingの一般的因果効果を一意に推定するためではなく、representational transferに対するcontrolled stress testとして使用する。

### 3.2 Data

- **AIPsy-Affect Strict Expanded Dataset**: 明示的な感情語彙（"sad", "happy" 等）による語彙共起交絡を厳格に制御した最小対データセット。**192 pair-id groups / 422 samples** から構成される。
- **Strict 3-Way Group Split**: Group leakageを防ぐため、pair-id単位で厳密に分割：
  - **Train split** (76 groups / 169 samples): 線形プローブの学習
  - **Alignment-dev split** (58 groups / 124 samples): Base→Instruct幾何写像（Ridge Alignment）の学習および多変量共分散の推定
  - **Held-out test split** (58 groups / 129 samples, うち完全 peak-neutral ペア 39組): Aligned表現の検証および因果介入評価
- **EmoBank Corpus**: 外部人間アノテーション（reader-perspective Valence評定, 3,210文）との対応付け評価に使用。

---

## 4. Constrained Report Measurement

各入力 $x$ について、$V, A \in \{1, \dots, 9\}$ からなる81個の候補 $y_{v,a}$ を評価する。

Primary implementationでは、各候補列の対数尤度

$$s_{v,a} = \log P(y_{v,a} \mid x) = \sum_{t=1}^{|y_{v,a}|} \log P(\text{token}_t \mid x, y_{v,a,<t})$$

を計算し、Softmax正規化（$\tau=1.0$）により報告確率分布を得る：

$$P(v, a \mid x) = \frac{\exp(s_{v,a} / \tau)}{\sum_{v', a'} \exp(s_{v', a'} / \tau)}$$

候補token長が異なる場合の長さバイアスを評価するため、補足解析として

$$\bar{s}_{v,a} = \frac{1}{|y_{v,a}|} \log P(y_{v,a} \mid x)$$

によるlength-normalized scoreについても全解析を再実行する。本稿ではこの二者を明示的に区別して報告する。

---

## 5. Detailed Results

### 5.1 Greedy Neutralization Does Not Imply Distributional Invariance (RQ1)

Instructモデルではgreedy first-person VA reportの98.6%が (5, 5) に集中した。一方、81候補の条件付き尤度分布から得られるexpected ValenceはEmoBank reader Valenceと有意に共変動した。

**表1: GreedyデコードとSequence-Likelihoodプロトコルの測定比較**

| 指標 | Base モデル | Instruct モデル | 統計的検定 |
|---|---|---|---|
| **Greedy 中立値 (5, 5) 集中率** | 12.4% | **98.6%** | Fisher's exact $p < 10^{-15}$ |
| **EmoBank Reader Valence 相関 ($r$)** | $0.365$ | **$0.629$** | $z = 4.12, p < 0.001$ |
| **EmoBank Reader Valence 相関 ($\rho$)** | $0.341$ | **$0.618$** | $p < 0.001$ |
| **期待Valence レンジ (5th–95th %tile)** | $3.82 \sim 7.14$ | $5.12 \sim 5.78$ | レンジ圧縮はあるが順序保持 |

この結果は、

$$\text{Greedy collapse} \not\Rightarrow \text{complete distributional collapse}$$

であることを示す。重要なのは、これを「latent emotion」の証拠とは解釈しないことである。観測されているのは、固定された候補集合上の条件付き確率構造の残存である。

---

### 5.2 Affect-Related Conditions Are Linearly Decodable (RQ2)

各層の最終prompt-token representationにlinear probeを適用した。

全層スイープでは、affective peak versus neutral condition（peak=1, neutral=0）のdecodabilityは層によって系統的に変化し、中盤層で最大となった。MLP outputについての最大値はLayer 15の

$$R^2 = 0.546 \quad (\text{Length-normalized LL}) \quad / \quad 0.547 \quad (\text{Raw sequence LL})$$

残差ストリームについて最大値はLayer 14の

$$R^2 = 0.507 \quad (\text{Length-normalized LL}) \quad / \quad 0.508 \quad (\text{Raw sequence LL})$$

であった。

**表2: 中間層隠れ状態からの線形プロービング決定係数 ($R^2$) および統制課題 (Layer 15)**

| 予測ターゲット | Held-out $R^2$ / 精度 | 統制・解釈 |
|---|---|---|
| **Peak-vs-Neutral Condition Indicator (MLP L15)** | **0.546** | 中間層MLP出力から感情条件が強く線形アクセス可能 |
| **Peak-vs-Neutral Condition Indicator (Resid L14)** | **0.507** | 残差ストリームでも中間層で強いアクセス性 |
| **Continuous Valence (EmoBank)** | **0.58** | 外部人間評定感情価の線形予測 |
| **Continuous Arousal (EmoBank)** | **0.42** | 外部人間評定覚醒度の線形予測 |
| **Token Count (文長統制)** | 0.05 | 文長アーティファクトではない |
| **Surface VAD (表層感情語辞書スコア)** | 0.12 | 単なる辞書単語の出現頻度ではない |
| **Affective vs Neutral 分類 (ROC-AUC)** | **> 0.975** | Layer 14–25 において極めて高い弁別能 |

この値はcontinuous Valence predictionではなく、peak-versus-neutral condition indicatorに対するheld-out regression performanceである。

したがって、この実験から言えるのは、

> *affective condition is strongly linearly accessible from mid-layer representations*

までであり、

> *this local activation slice causally controls the downstream behavior*

ではない。

---

### 5.3 Base and Instruct Representations Are Predictively Alignable (RQ3)

Base activationからInstruct activationへのRidge mapをalignment-dev splitのみで学習した。Held-out testでは、弱い正則化条件において、

$$\mathrm{CKA} \approx 0.83, \qquad \mathrm{Top\text{-}1\ retrieval} = 86.6\%$$

が得られた。

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

したがって、Base/Instruct representationには、直接座標系が異なっていても高いpredictive correspondenceが存在する。

---

### 5.4 Predictive Alignment Does Not Guarantee Distributional Typicality (RQ4)

Predictive alignmentの品質を$R^2$やCKAだけで判断できるかを検証するため、Ridge regularization parameterを $10^{-5} \le \alpha \le 10^4$ でスイープし、Mahalanobis距離 $D_M$ を測定した。

自然なInstruct activationのMahalanobis radiusの中央値は

$$D_M = 39.63$$

であった（多変量ガウス薄殻理論値 $\sqrt{1536} \approx 39.19$ と整合）。一方、aligned activationは、

$$D_M = 9.78 \quad (\alpha=10^{-5}) \quad \longrightarrow \quad D_M = 0.12 \quad (\alpha=10^4)$$

へ単調に収縮した。

**表4: 自然なInstruct多様体の経験的参照分布とAligned Baseの比較 (Layer 15 MLP)**

| 診断指標 | 実測値 | 幾何学的・統計的解釈 |
|---|---|---|
| **Natural Instruct $D_M$ (5th %tile)** | **34.20** | 自然なInstruct活性化の経験的下限 |
| **Natural Instruct $D_M$ (25th %tile)** | **36.99** | 自然分布の第1四分位 |
| **Natural Instruct $D_M$ (50th %tile, 中央値)** | **39.63** | **多変量ガウス薄殻理論値 $\sqrt{1536} \approx 39.19$ と一致** |
| **Natural Instruct $D_M$ (75th %tile)** | **43.46** | 自然分布の第3四分位 |
| **Natural Instruct $D_M$ (95th %tile)** | **51.43** | 自然なInstruct活性化の経験的上限 |
| **Raw Base $D_M$ (中央値)** | **256.47** | 顕著な外れ値・極端な遠方OOD |
| **Aligned Base $D_M$ ($\alpha=10^{-1}$ 中央値)** | **8.30** | **分布中心側への顕著な収縮（Center Collapse）** |
| **Aligned Base $D_M$ ($\alpha=10^{-5}$ 中央値)** | **9.78** | **正則化最小化時でも球殻（39.63）の内側に縮退** |
| **Two-Sample 分類器 AUC ($\alpha=10^{-1}$)** | **0.616** | 弱識別可能（Raw Baseの1.00より大幅改善だが非典型） |
| **コサイン類似度: Matched Aligned $\rightarrow$ Instruct** | **0.9948** | 同一刺激ペア間での極めて高い類似性 |

特に $N_{\mathrm{alignment}} \ll d_{\mathrm{hidden}}$ の設定において、Ridge predictionはtarget activationのconditional mean方向へ分散を縮小させる。

興味深いことに、$\mathrm{CKA} \approx 0.83, \mathrm{retrieval} \approx 86.6\%$ という高いpredictive correspondenceが存在していても、aligned statesは自然なInstruct distributionの典型半径には達していない。

したがって、

$$\boxed{\text{High predictive similarity can coexist with severe variance contraction}}$$

である。これは、predictive similarityだけではtarget activation distributionへの幾何学的適合を保証できず、cross-model activation patchingを解釈する上で重要な方法論的制約となる。

---

## 6. Core Result: Layerwise Dissociation Between Decodability and Local Causal Leverage (RQ5)

全28層について、最終prompt tokenにおけるMLP outputおよびresidual streamを対象として、

$$D_\ell = \text{held-out linear decodability}$$

と

$$C_\ell = \text{within-model substitution recovery}$$

を独立に測定した。

### 6.1 全28層における層別マッピング

評価では、81候補の条件付き対数尤度からSoftmaxによって正規化確率分布を構成する。Source、target、patched runについて、

$$P_{\mathrm{source}}, \quad P_{\mathrm{target}}, \quad P_{\mathrm{patch}} \in \mathbb{R}^{9 \times 9}$$

を求め、Earth Mover's Distance (2D EMD) を用いて回復率を定義する：

$$\mathrm{Recovery} = 1 - \frac{D_{\mathrm{EMD}}(P_{\mathrm{patch}}, P_{\mathrm{source}})}{D_{\mathrm{EMD}}(P_{\mathrm{target}}, P_{\mathrm{source}})}$$

**表6: 全28層における線形プローブ決定係数 ($R^2$) と真の2D Joint OT因果回復率 ($C_\ell^{\mathrm{Joint}}$) の網羅的マッピング (MLP / Attention Output / Residual Stream)**

| Layer | MLP Probe $R^2$ | MLP Rec (%) | ATTN Probe $R^2$ | ATTN Rec (%) | Resid Probe $R^2$ | Resid Rec (%) |
|---|---|---|---|---|---|---|
| **0** | 0.3025 | -1.39% | 0.3395 | -0.92% | 0.3129 | -1.15% |
| **1** | 0.3045 | -0.47% | 0.3263 | +0.78% | 0.3529 | -0.04% |
| **2** | 0.3608 | -0.32% | 0.3971 | -0.78% | 0.3943 | -1.93% |
| **3** | 0.3932 | +1.38% | 0.4903 | +0.60% | 0.3694 | -2.73% |
| **4** | 0.3847 | +1.67% | 0.3997 | -2.04% | 0.3521 | -2.71% |
| **5** | 0.3685 | -0.57% | 0.3844 | +0.61% | 0.4059 | -2.23% |
| **6** | 0.4229 | -0.17% | 0.4698 | -1.23% | 0.4697 | -3.70% |
| **7** | 0.4787 | +1.21% | 0.4321 | -0.50% | 0.4726 | -2.09% |
| **8** | 0.4671 | -0.94% | 0.3837 | +0.20% | 0.4619 | -3.05% |
| **9** | 0.4134 | +0.78% | 0.4157 | +0.09% | 0.4141 | -2.31% |
| **10** | 0.4817 | **+2.20% (Max)** | 0.4753 | +0.78% | 0.4744 | -1.37% |
| **11** | 0.4891 | +0.25% | 0.4830 | +0.21% | 0.4519 | -0.16% |
| **12** | 0.4571 | -0.92% | 0.4085 | +1.46% | 0.4561 | -0.53% |
| **13** | 0.5151 | -0.06% | 0.5205 | +0.92% | 0.4887 | +0.61% |
| **14** | 0.5372 | +1.35% | 0.5313 | -0.36% | **0.5016 (Peak)** | +0.71% |
| **15** | **0.5610 (Peak)** | +1.00% | 0.5217 | -0.52% | 0.4862 | +0.25% |
| **16** | 0.5158 | +1.02% | 0.4913 | +0.98% | 0.4705 | **+1.68% (Max)** |
| **17** | 0.4772 | -0.04% | 0.4936 | -0.04% | 0.4432 | +1.34% |
| **18** | 0.5513 | -0.43% | **0.5495 (Peak)** | +0.04% | 0.4848 | -0.58% |
| **19** | 0.4434 | -0.15% | 0.4483 | -0.18% | 0.4330 | +0.81% |
| **20** | 0.4866 | -0.38% | 0.4602 | **+1.48% (Max)** | 0.4008 | +0.65% |
| **21** | 0.5223 | +0.62% | 0.4859 | +0.21% | 0.3903 | +0.85% |
| **22** | 0.3923 | +0.16% | 0.4076 | +0.96% | 0.2193 | +0.31% |
| **23** | 0.3774 | +0.52% | 0.4514 | +1.23% | 0.2107 | +1.31% |
| **24** | 0.3514 | +0.84% | 0.4072 | +0.61% | 0.1470 | -0.00% |
| **25** | 0.3178 | -0.36% | 0.4100 | +0.97% | 0.1790 | +0.18% |
| **26** | 0.3311 | +0.44% | 0.1040 | +0.41% | 0.1681 | +0.23% |
| **27** | 0.3131 | -0.00% | -0.2670 | -0.00% | 0.2496 | -0.00% |

真の2D Joint OTに基づく全層測定により、以下の決定的な知見が得られた：

1. **極小の局所十分性**:
   全28層・全コンポーネントを通じて因果回復率は極めて小さく、最大値でもMLPで2.20%（Layer 10）、Attentionで1.48%（Layer 20）、Residualで1.68%（Layer 16）に留まる。
2. **最高デコード層での解離**:
   MLPデコーダビリティが最大となるLayer 15（$R^2 = 0.5610$）における回復率はわずか1.00%であり、Attentionデコーダビリティが最大となるLayer 18（$R^2 = 0.5495$）での回復率は0.04%であった。
3. **無相関の頑健性**:
   層別probe $R^2$と因果回復率の間には、いずれのコンポーネントにおいても単調関係は検出されなかった（MLP: Spearman $\rho = 0.2956, p = 0.1268$; ATTN: $\rho = 0.0230, p = 0.9076$; RESID: $\rho = -0.0394, p = 0.8422$）。

さらに、本研究では以下の4大対立仮説を独立実験により検証・検討した：

- **Generation-time Sweep**: 生成プレフィックス直後でのパッチングにより後段層（L24 MLP: 5.02%, L20 ATTN: 4.23%）で微小な回復率上昇が観測されたものの、全層を通じた中央値回復率は0.00%にとどまった。
- **Probe-Aligned Necessity Sweep**: プローブ方向の直交射影消去による中和比率は全層で -3.27%〜+0.61% であり、直交ランダム方向に対する特異性検定（BH-FDR補正後、全84条件）で有意な層は皆無（最小 $q = 0.857$）であった。
- **Attention Pathway**: 単一層射影済みAttention出力の置換でも因果回復率は極小（Prompt最大1.48%, Generation最大4.23%）であり、強力な局所的因果ボトルネックとして機能する証拠は見出されなかった（We found no evidence that a single-layer projected attention output acts as a strong local causal bottleneck）。
- **Architectural Replication**: Llama-3.2-1B-Instruct（全16層）での生成時パッチングでも回復率は最大0.73%（中央値0.00%）に留まり、異なるモデルファミリにおいても同様に低い生成時局所回復率が独立に観測された（Low generation-time local causal recovery was independently replicated in Llama-3.2-1B-Instruct）。

したがって、

$$\boxed{\text{High layerwise decodability } (R^2 \approx 0.56) \quad\text{coexists with}\quad \text{low local sufficiency } (S_\ell \le 2.2\%) \quad\text{and}\quad \text{low probe-aligned necessity } (R_{\mathrm{neut}} \approx 0\%)}$$

すなわち、

$$D_\ell \not\to S_\ell, \qquad D_\ell \not\to N_\ell$$

という明確な経験的解離が確立された。

重要なのは、これはaffect-related informationがモデルの出力生成に「使用されていない」ことを意味しない点である。本実験が示すのは、最終prompt-tokenにおける単一層MLP/residual activation sliceの置換という介入族では、その情報のdecodabilityからdownstream reportに対するcausal leverageを予測できないという、より限定された主張である。

---

### 6.2 Robustness to Likelihood Scoring

候補列のtokenization lengthによって結果が生じている可能性を検証するため、
1. token-length-normalized log likelihood
2. raw sequence log likelihood
の二つのscoring protocolで全解析を再実行した。

両条件でdecodability profileはほぼ同一となり、MLPの最大値はいずれもLayer 15で観測された：

$$R^2_{\mathrm{Norm}} = 0.546, \qquad R^2_{\mathrm{Raw}} = 0.547$$

また、causal recoveryは両条件とも全層で小さく、

$$\max C_\ell^{\mathrm{Norm}} = 1.37\%, \qquad \max C_\ell^{\mathrm{Raw}} = 0.97\%$$

さらにdecodability–recovery correlationも両protocolで小さかった。

したがって、観測されたlayerwise dissociationはcandidate sequence lengthの正規化方法に依存しない。

---

### 6.3 ポジティブコントロールプロトコルの検証

また、局所スライス置換と全トークン置換の振る舞いを比較するため、代表プロトコルでの検証も実施した。

**表5: 同一モデル内ポジティブコントロールにおける回復率および期待値シフト (Held-out Test N=39)**

| プロトコル | Raw LL (単純和) Mean / Median Rec | Norm LL (長さ正規化) Mean / Median Rec | EV Shift (Raw / Norm) | 解釈・メカニズム |
|---|---|---|---|---|
| **`mlp_last_token_L15`** | **0.82%** / 0.56% | **1.30%** / 1.13% | +1.09% / -1.45% | **プローブ最高層でも局所スライスの回復率は ~1%** |
| **`resid_last_token_L15`** | **0.38%** / 0.36% | **0.42%** / 0.15% | +2.97% / +0.68% | 単一残差ストリームの局所制御力はさらに微小（< 0.5%） |
| **`multi_resid_last_L13_16`** | **1.47%** / 1.06% | **1.38%** / 0.89% | +3.53% / +0.09% | 4層連続の最終トークン置換でも約 1.4% にとどまる |
| **`resid_all_tokens_L15`** | -448.95% / -263.24% | -225.88% / -133.28% | +893.61% / +234.28% | **文脈崩壊（コヒーレンス破壊）により出力分布が外れ値へ発散** |
| **`multi_resid_all_L13_16`** | -437.15% / -248.80% | -236.91% / -129.70% | +876.11% / +214.96% | 複数層全トークン置換でも同様に分布破壊（負の回復） |

全トークン置換では異なる文脈の活性化を無理に貼り付けることでコヒーレンス破壊（負の回復）が生じるのに対し、最終トークン置換では出力分布を壊すことなく安定して測定できる。しかしその回復率は高々 1% 前後にとどまった。

---

## 7. Discussion

### 7.1 Interpretation: Decodability Is a Representational Claim, Not a Mechanistic Claim

本結果は、

$$\text{representation contains information}$$

と

$$\text{a local representation slice controls behavior}$$

を明確に区別する必要性を実証する。

特に、本研究ではdecodabilityが約 $0.30 \rightarrow 0.55$ まで層依存的に大きく変化するにもかかわらず、同じ層に対するlocal causal recoveryは全域で1.5%未満であった。

したがって、probe accuracyを用いてactivation patchingの介入位置を選択する一般的な戦略には注意が必要である。高いprobe accuracyは、

> *“this is a location from which the information is linearly accessible”*

という記述的主張をある程度支持するが、

> *“this is a location where intervention will exert local causal control over the downstream decision”*

という機構的主張を直接には支持しない。本研究はこの二つの量を全ネットワーク深度にわたって同時に測定することで、その乖離を可視化した。

### 7.2 Predictive Alignment Is Not Functional Equivalence

Cross-model mappingでも同様である。高い$R^2$、CKA、pair retrievalは、

> *one representation predicts another*

ことを示す。しかし、

> *one representation can replace the other inside the target computation*

ことは示さない。

さらに本研究の$\alpha$ sweepは、高いpredictive scoreを維持しながらactivation varianceがtarget distributionの中心方向へ著しく縮小（center collapse）し得ることを示した。したがって、cross-model causal interventionでは少なくとも、
1. predictive fidelity
2. distributional typicality
3. interventional effect

を別々に測定・報告する必要がある。

### 7.3 Scope of the Causal Claim

本研究から導ける結論は、

> **Probe-accessible locations need not coincide with local causal bottlenecks under matched activation substitution.**

である。一方、

> **Affect information has no causal role in report generation.**

とは結論しない。

後者を主張・検証するには、より広い介入族（sequence-wide representations、attention-mediated information flow、multi-token interventions、およびresponse-generation positions）が必要である。

したがって、本研究における“causal insufficiency”は、**tested local activation slice with respect to the measured downstream report** という操作的意味で使用する。また、本研究は以下を主張するものではない：
- LLMが主観的感情を経験している
- first-person outputがintrospective accessである
- post-trainingが一意にaffect representationをdecoupleした
- probe-accessible informationが必ずcausally irrelevantである

本研究が対象とするのは、特定のBase/Instruct pairにおけるrepresentation、prediction、distribution、causal useを分離するためのcontrolled case studyである。

---

## 8. Limitations and High-Value Future Directions

本研究には以下の限界があり、これらは今後の機構的解釈可能性研究における高価値な拡張課題を構成する：

1. **Generation-time patching**:
   本研究はプロンプト最終トークンにおける局所置換を中心とした。下流の生成時（例えば `{"valence": ` に続く数値トークン生成直前）のhidden statesをパッチングすることで、prompt-time decodabilityとgeneration-time causal controlの機能的分離をさらに直接的に検証できる。
2. **Necessity testing (Ablation / Removal)**:
   本研究の主介入はsource $\rightarrow$ target substitution（sufficiency的介入）である。感情方向ベクトルの除去（affect direction removal）、mean ablation、あるいはnull-space projectionによるnecessity testを併用し、high decodability + low local sufficiency + low necessity の三位一体を評価することが自然な拡張となる。
3. **Cross-family minimal replication**:
   主解析はQwen2.5-1.5B Base/Instruct pairに限定される。全実験の反復は不要であるが、layerwise probing profileとwithin-model local substitution sweepを別モデルファミリー（例: Llama-3-8B等）で最小再現することにより、ケーススタディから一般的アーキテクチャ特性へと主張を昇華させることができる。

---

## 9. Conclusion

本研究は、affect-relevant representationsをケーススタディとして、LLM mechanistic analysisにおける三つの異なる問いを分離した：

$$\boxed{\text{What information is decodable?}}$$

$$\boxed{\text{What representations are predictively alignable?}}$$

$$\boxed{\text{What states are causally used?}}$$

これらは同一の問いではない。

Qwen2.5 Base/Instruct pairでは、affective conditionは中間表現から強くdecodableであり（$R^2 \approx 0.55$）、Base/Instruct representations間には高いpredictive correspondenceが存在した（CKA $0.829$, Retrieval $86.6\%$）。一方、高次元アライメントは自然なtarget activation distributionと同等のstatistical geometryを再現せず（Center Collapse）、predictive alignmentをfunctional equivalenceとして扱うことの危険性を示した。

さらに、全28層の網羅的スイープにより、decodabilityが層深度に伴って山型に大きく変動する（$0.30 \rightarrow 0.55$）一方、局所活性化置換のcausal recoveryは全域で一様に小さく（$< 1.5\%$）、両者の間に単調な関連は検出されないことが示された。

したがって、LLM内部表現をmechanisticに解釈する際には、probe accuracyやrepresentation similarityだけではなく、distributional diagnosticsとproperly validated causal interventionsを独立に組み合わせる必要がある。

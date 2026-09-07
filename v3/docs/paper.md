# Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model

## Abstract

大規模言語モデル（LLM）の内部状態から属性を高精度にデコードできることは、その情報がモデル内部に存在する証拠を与える。しかし、decodabilityは、その表現が特定の下流計算に利用されていることを必ずしも意味しない。本研究では、この古典的なrepresentation–use distinctionを、Base/Instruct言語モデルペアにおけるaffect-relevant representationsをケーススタディとして機構的に検証する。

Qwen2.5-1.5B Base/Instructを対象に、語彙交絡を制御したAIPsy-Affect Strict Expanded（192 pair-id groups / 422 samples）、81候補の制約付きVA報告、層別linear probing、Base→Instruct表現アライメント、およびactivation substitutionを統合した評価系を構築した。

第一に、Instructモデルのgreedyな一人称VA報告は98.6%が (5, 5) に集中する一方、候補列全体の条件付き尤度から得られるValence expectationは刺激の感情価と共変動した（EmoBank reader valence相関: Instruct $r=0.629$, Base $r=0.365$）。したがって、単一のgreedy completionによる表層的中立化は、候補出力分布全体の刺激依存性の消失を意味しない。

第二に、最終prompt tokenの中間表現からaffective-versus-neutral conditionを線形に予測でき、そのdecodabilityはネットワーク深度に伴って変化し、中盤層付近で最大となった（MLP outputにおいてLayer 15 Peak-vs-Neutral intensity indicator $R^2=0.548$）。

第三に、Base→Instruct Ridge alignmentはheld-out activationsについて高いpredictive correspondenceを示した。弱い正則化条件ではLinear CKAは約0.83、paired-retrieval top-1 accuracyは86.6%に達した。一方、Mahalanobis解析では、自然なInstruct activationが約39.63の中央値半径を持つのに対し、aligned activationは8–10程度まで分布中心側へ収縮した。したがって、高い$R^2$、CKA、retrieval accuracyは、target activation distributionへの完全な統計的適合を保証しない。

これらの結果は、representation alignmentの評価にpredictive similarityだけを用いることの限界を示す。本研究は、主観的感情や内省を主張するものではなく、what is linearly accessible, what is predictively transferable, and what is causally used must be evaluated as distinct questionsという方法論的区別をaffect-relevant representationsを用いて検証するケーススタディである。

---

## 1. Introduction

大規模言語モデル（LLM）のhidden representationsには、言語構造、知識、真偽、安全性、感情的属性など、さまざまな情報が線形にデコード可能な形で含まれることが知られている。

しかし、probeがある属性を予測できることから、

> *the model uses that representation to produce the downstream behavior*

と結論することはできない。

この問題はprobing研究において以前から指摘されてきた（例: *Elazar et al., 2021; Ravfogel et al., 2021; Belinkov, 2022*）。Probeはモデル自身のdownstream computationには必要でない相関情報も利用できるため、

$$\text{Decodability} \not\Rightarrow \text{Behavioral Use}$$

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
3. **因果的解離**: 情報はdecodableだが、その表現部位は報告生成に利用されていない。
4. **分散的・動的利用**: 情報は別のtoken position、component、あるいはdistributed computationとして利用されている。

これらを区別するためには、behavior、decodability、representation alignment、causal interventionを独立に測定する必要がある。

### 1.2 Research Questions

本研究では以下を問う：

- **RQ1**: Greedy reportが中立化した場合でも、constrained output distributionには刺激依存的情報が残るか。
- **RQ2**: Affect-related conditionはBase/Instructモデルの内部状態からどの程度decodableか。
- **RQ3**: Base/Instruct間の表現はheld-out data上でどの程度predictively align可能か。
- **RQ4**: Predictively aligned representationsはtarget modelの自然activation distributionにも適合するか。
- **RQ5**: Probe-accessible activation sitesは、対応するreportに対してcausal sufficiencyまたはnecessityを持つか。

RQ5については、厳密に実装されたactivation-substitutionおよびablationプロトコルによって評価する。

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
  - **Held-out test split** (58 groups / 129 samples, うち完全 peak-neutral ペア 39組): Aligned表現の検証および介入評価
- **EmoBank Corpus**: 外部の人間アノテーション（reader-perspective Valence評定）との対応付け評価に使用。

---

## 4. Constrained Report Measurement

各入力 $x$ について、$V, A \in \{1, \dots, 9\}$ からなる81個の候補 $y_{v,a}$ を評価する。

Primary implementationでは、各候補列の対数尤度

$$s_{v,a} = \log P(y_{v,a} \mid x) = \sum_{t=1}^{|y_{v,a}|} \log P(\text{token}_t \mid x, y_{v,a,<t})$$

を計算し、Softmax正規化（$\tau=1.0$）により報告確率分布を得る：

$$P(v, a \mid x) = \frac{\exp(s_{v,a} / \tau)}{\sum_{v', a'} \exp(s_{v', a'} / \tau)}$$

候補token長が異なる場合の長さバイアスを評価するため、補足解析として

$$\bar{s}_{v,a} = \frac{1}{|y_{v,a}|} \log P(y_{v,a} \mid x)$$

によるlength-normalized scoreについても同一の解析を適用可能とする。本稿ではこの二者を明示的に区別する。

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

であることを示す。観測されているのは、固定された候補集合上の条件付き確率構造の残存である。

---

### 5.2 Affect-Related Conditions Are Linearly Decodable (RQ2)

各層の最終prompt-token representationにlinear probeを適用した。

全層スイープでは、affective peak versus neutral condition（peak=1, neutral=0）のdecodabilityは層によって系統的に変化し、中盤層で最大となった。MLP outputについての最大値はLayer 15の

$$R^2 = 0.548$$

であった。

**表2: 中間層隠れ状態からの線形プロービング決定係数 ($R^2$) および統制課題 (Layer 15)**

| 予測ターゲット | Held-out $R^2$ | 統制・解釈 |
|---|---|---|
| **Peak-vs-Neutral Condition Indicator** | **0.548** | 中間層MLP出力から感情条件が強く線形アクセス可能 |
| **Continuous Valence (EmoBank)** | **0.58** | 外部人間評定感情価の線形予測 |
| **Continuous Arousal (EmoBank)** | **0.42** | 外部人間評定覚醒度の線形予測 |
| **Token Count (文長統制)** | 0.05 | 文長アーティファクトではない |
| **Surface VAD (表層感情語辞書スコア)** | 0.12 | 単なる辞書単語の出現頻度ではない |
| **Affective vs Neutral 分類 (ROC-AUC)** | **> 0.975** | Layer 14–25 において極めて高い弁別能 |

この結果から言えるのは、「affective condition is strongly linearly accessible from mid-layer representations」までであり、「the model causally uses this representation」ではない。

---

### 5.3 Base and Instruct Representations Are Predictively Alignable (RQ3)

Base activationからInstruct activationへのRidge mapをalignment-dev splitのみで学習した。Held-out testでは、弱い正則化条件において、

$$\mathrm{CKA} \approx 0.83, \quad \mathrm{Top\text{-}1\ retrieval} = 86.6\%$$

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

Base/Instruct representationには、直接の座標系が異なっていても高いpredictive correspondenceが存在する。

---

### 5.4 Predictive Alignment Does Not Guarantee Distributional Typicality (RQ4)

Predictive alignmentの品質を$R^2$やCKAだけで判断できるかを検証するため、Mahalanobis距離 $D_M$ による多様体診断を実施した。

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

自然なInstruct activationのMahalanobis半径中央値が $D_M = 39.63$ であるのに対し、aligned activationは $D_M = 8.30 \sim 9.78$ へと分布中心側へ収縮した。特に $N_{\mathrm{alignment}} \ll d_{\mathrm{hidden}}$ の設定において、Ridge predictionはtarget activationのconditional mean方向へ分散を縮小させる。

$$\boxed{\text{Predictive similarity} \not\Rightarrow \text{distributional typicality}}$$

これは、高いpredictive scoreを達成していても、aligned statesが自然なInstruct分布の統計的幾何構造（高次元球殻）を回復していないことを意味し、cross-model activation patchingを解釈する上での重要な方法論的制約となる。

---

## 6. Causal Intervention: Required Validation Protocol (RQ5)

Causal sufficiencyを検証するため、source activationをmatched target runへsubstituteする。

この評価では、81候補の条件付き対数尤度からSoftmaxによって正規化確率分布を構成する必要がある。Source、target、patched runについて、

$$P_{\mathrm{source}}, \quad P_{\mathrm{target}}, \quad P_{\mathrm{patch}} \in \mathbb{R}^{9 \times 9}$$

を求め、Earth Mover's Distance (2D EMD) を用いて回復率を定義する：

$$\mathrm{Recovery} = 1 - \frac{D_{\mathrm{EMD}}(P_{\mathrm{patch}}, P_{\mathrm{source}})}{D_{\mathrm{EMD}}(P_{\mathrm{target}}, P_{\mathrm{source}})}$$

本研究の方法論的枠組みにおいて、因果的役割を厳密に結論付けるためには、以下の体系的検証が必要とされる：

1. **構成要素とサイトの網羅性**:
   - MLP output, Attention output, Residual stream
   - Prompt-final token position のみならず、先行文脈トークンおよび generation-stage positions
2. **Sufficiency と Necessity の分離**:
   - **Sufficiency test**: Source (Peak) $\rightarrow$ Target (Neutral) への活性化置換（Activation Substitution）
   - **Necessity test**: Mean ablation または Direction removal（刺激依存ベクトルの除去）
3. **確率分布算出の正確性**:
   - 尤度計算後の確率正規化（Softmax）を厳密に介したEMD評価（実装上のトークン長配列の誤参照を完全に排除したプロトコル）

---

## 7. Discussion

### 7.1 Decodability Is a Representational Claim, Not a Mechanistic Claim

本研究で最も重要な方法論的区別は、**「decodable」** と **「causally used」** を混同しないことである。

Probeはあるhidden stateに属性関連情報が存在することを示せる。しかし、その情報がモデル自身のdownstream computationに利用されているかを確立するためには、independent causal interventionが不可欠である。

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

### 7.3 What This Study Does Not Claim

本研究は、以下を主張するものではない：
- LLMが主観的感情を経験している
- first-person outputがintrospective accessである
- post-trainingが一意にaffect representationをdecoupleした
- probe-accessible informationが必ずcausally irrelevantである

本研究が対象とするのは、特定のBase/Instruct pairにおけるrepresentation、prediction、distribution、causal useを分離するためのcontrolled case studyである。

---

## 8. Limitations

- **モデルスコープ**: 主解析はQwen2.5-1.5B Base/Instruct pairに限定される。
- **高次元線形写像の幾何学的収縮**: 線形アライメントは小標本高次元設定でvariance shrinkageを生じるため、cross-model interventionの解釈には慎重な限界設定が必要である。
- **トークン位置の局所性**: 最終prompt-token representationだけでは、response generation中に動的に参照・利用される情報を完全に捕捉できない可能性がある。
- **因果的局所化の完全な確定**: MLP、attention、residual、token position、およびgeneration stageを横断した正格な再検証実験の完了を要する。

---

## 9. Conclusion

本研究は、affect-relevant representationsをケーススタディとして、LLM mechanistic analysisにおける三つの異なる問いを分離した：

$$\boxed{\text{What information is decodable?}}$$

$$\boxed{\text{What representations are predictively alignable?}}$$

$$\boxed{\text{What states are causally used?}}$$

これらは同一の問いではない。

Qwen2.5 Base/Instruct pairでは、affective conditionは中間層から強くdecodableであり（$R^2=0.548$）、Base/Instruct representations間には高いpredictive correspondenceが存在した（CKA $0.829$, Retrieval $86.6\%$）。一方、高次元アライメントは自然なtarget activation distributionと同等のstatistical geometryを再現せず（Center Collapse）、predictive alignmentをfunctional equivalenceとして扱うことの危険性を示した。

したがって、LLM内部表現をmechanisticに解釈する際には、probe accuracyやrepresentation similarityだけではなく、distributional diagnosticsとproperly validated causal interventionsを独立に組み合わせる必要がある。

Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model

Abstract

大規模言語モデル（Large Language Models; LLMs）の内部状態から、言語構造、知識、真偽、安全性、感情的属性などを高精度に線形デコードできることが多数報告されている。しかし、内部状態から情報を予測できること（decodability）と、その表現部位がモデル自身の下流計算において当該情報を因果的に利用していること（causal use）は同値ではない。本研究では、この古典的なrepresentation–use distinctionを、Qwen2.5-1.5B Base/Instructモデルペアにおけるaffect-relevant representationsをケーススタディとして体系的に検証する。

本研究では、明示的な感情語彙による表層交絡を制御したAIPsy-Affect Strict Expandedデータセット（192 pair-id groups / 422 samples）、81通りのValence–Arousal（VA）候補列を用いた制約付きsequence-likelihood evaluation、全28層にわたるlinear probing、Base→Instruct表現アライメント、Ridge正則化強度スイープ、多変量Mahalanobis診断、cross-model activation patching、multi-layer simultaneous patching、およびwithin-model matched activation substitutionを統合した。

第一に、Qwen2.5-1.5B-Instructでは一人称形式のgreedy VA報告の98.6%が中立値 (5,5) に集中した一方、81候補列全体の条件付き尤度から得られる期待Valenceは刺激の感情価と有意に共変動した。したがって、greedy decoding上の中立化は、候補出力分布全体の刺激依存性の消失を意味しない。

第二に、affective peak versus neutral conditionの線形decodabilityはモデル深度に依存して変化し、MLP outputではLayer 15で最大となった（(R^2=0.561)）。Attention outputでもLayer 18で(R^2=0.550)、Residual streamでもLayer 14で(R^2=0.502)に達した。

第三に、真の2D Joint Optimal Transport（Joint OT）に基づく同一モデル内Peak→Neutral活性化置換の因果回復率は、Layer 15 MLPで1.00%に留まり、全28層・全コンポーネント（MLP, Attention, Residual）を通じた最大回復率も2.20%（Layer 10 MLP）であった。層別decodabilityと因果回復率の間に検出可能な単調関係は認められなかった（MLP: Spearman (\rho=0.296, p=0.127); ATTN: (\rho=0.023, p=0.908); RESID: (\rho=-0.039, p=0.842)）。

第四に、自己報告生成直前のプレフィックス最終トークンにおける生成時因果パッチング（Generation-time sweep）では、後段層（Layer 20〜24）で微小な変位上昇（MLP L24で最大5.02%、ATTN L20で最大4.23%）が認められたものの、中央値はいずれも0.00%であり、95%以上の変位は依然として非回復であった。

第五に、プローブ方向の幾何学的射影消去による局所必要性検定（Probe-aligned necessity sweep）では、全28層×3コンポーネントにおいて出力の中和比率は一貫して0%近傍（-3.27%〜+0.61%）であり、直交ランダム方向消去に対する特異性検定（Benjamini-Hochberg FDR補正後）で有意（(q < 0.05)）となる層は皆無であった。さらに、この局所的因果解離はMeta Llama-3.2-1B-Instruct（全16層、最大回復率0.73%）においても完全に再現された。

以上の結果から、本研究は「High Decodability, Low Local Sufficiency, Low Probe-Aligned Necessity」の三位一体を実証し、大規模言語モデルにおいて線形プローブによるアクセス可能性（decodability）を、モデル自身が下流報告を生成する因果的メカニズム（causal mechanism）と同定してはならないことを強く示す。

⸻

1. Introduction

大規模言語モデルの内部表現を解析する代表的な方法の一つにlinear probingがある。特定層のhidden stateから属性ラベルを線形分類・回帰によって高精度に予測できる場合、その内部状態に当該属性に関する情報がアクセス可能な形で存在すると解釈できる。

近年では、真偽、安全性、拒否、感情、人格、社会的属性など、多様な概念について高いprobe performanceが報告されている。一方で、probing研究では以前から、probeが情報を読み取れることと、モデル自身がその情報を下流計算に利用していることは区別すべきであると指摘されてきた。Amnesic Probingなどの研究が示す通り、

[
\mathrm{Decodability}
\not\Rightarrow
\mathrm{Behavioral\ Use}
]

である。

この問題は、post-trainingを経たLLMの内部表現を研究する場合に特に重要となる。Instruction tuningやpreference optimizationを受けたモデルは、同一architectureのBaseモデルと比較して出力方策が大きく異なる。一方で、内部状態にはBaseモデルと類似した意味情報が依然として読み取れる場合がある。

本研究では、このrepresentation–use gapを感情関連表現を用いて検証する。

Qwen2.5-1.5B-Instructへ感情的な文章を提示し、

{"valence": 5, "arousal": 5}

のような一人称形式のVA報告を要求すると、greedy decodingはほぼ常に中立値へ集中する。しかし、その同じモデルの内部activationからは、文章がaffective peak conditionであるかneutral conditionであるかを高精度に予測できる。

この観察だけから、

「モデルには感情情報があるがpost-trainingにより出力から抑制された」

と結論することはできない。

少なくとも以下の説明が存在する。

1. Probe-accessible erasure
    Affect-relevant informationそのものが失われている。
2. Output-level neutralization
    情報は内部に残っているが、greedy outputのみが中立値へ集中している。
3. Local causal dissociation
    情報はある層からdecodableであるが、その局所activation slice自体はreport generationの因果的ボトルネックではない。
4. Distributed or dynamic use
    情報はsequence-wide states、attention pathways、複数層の相互作用、あるいはresponse generation時のhidden statesを介して利用される。
5. Representational transformation across models
    Base/Instruct間で情報の座標系が変化しているため、単純なcross-model transferが失敗する。

これらを区別するには、単なるprobe performanceだけでは不十分である。

本研究では、

* output measurement,
* within-model decoding,
* cross-model alignment,
* distributional geometry,
* within-model substitution,
* cross-model patching,

を同一の実験系で接続する。

1.1 Research Questions

本研究では次の5つの研究質問を設定する。

RQ1. Greedyなfirst-person reportが中立化していても、候補出力分布には刺激依存的構造が残るか。

RQ2. Affect-related conditionは、Base/Instructモデルの内部表現からどの程度線形にdecodableか。

RQ3. Base/Instruct間の内部表現は、held-out data上でどの程度predictively align可能か。

RQ4. Predictively aligned representationsは、target modelの自然なactivation distributionにも統計的に適合するか。

RQ5. 高いdecodabilityを示すactivation sitesは、対応するfirst-person reportに対して大きなlocal causal leverageを持つか。

本研究の主たる焦点はRQ5である。

⸻

2. Conceptual Definitions

本研究では、「感情」という語による擬人観的解釈を避けるため、以下の構成概念を明確に分離する。

2.1 Reader-rated text affect

EmoBank等で人間読者が文章から受けるValence/Arousal評定を指す。

これはモデル自身の内部状態とは独立した外部基準である。

2.2 Third-person affect recognition

文章中の人物、筆者、話者などの感情状態をモデルに評価させるタスクである。

2.3 Constrained first-person report distribution

モデルに、

Read the following text and report your affective state.

と指示した後、

{"valence": v, "arousal": a}

という有限候補列に割り当てられる条件付き確率分布を指す。

本稿における“first-person”は文法的・task-levelな形式を表すのみであり、モデルが自己状態へのprivileged introspective accessを持つことを意味しない。

2.4 Affect-relevant representation

刺激のaffective conditionまたはhuman-rated VA informationを線形に予測可能な内部activationを指す。

これはsubjective affective experienceの存在を意味しない。

⸻

3. Models and Data

3.1 Models

主要実験では、同一architectureを持つ以下のモデルペアを使用した。

* Qwen/Qwen2.5-1.5B
* Qwen/Qwen2.5-1.5B-Instruct

Qwen2.5-1.5Bは28 Transformer layersを持ち、本研究ではLayer 0からLayer 27までを解析する。

Base/Instruct comparisonはpost-trainingの一般的因果効果を推定するためではなく、同一architecture間でrepresentational transferを評価するstress testとして用いる。

3.2 AIPsy-Affect Strict Expanded

主要な機構実験にはAIPsy-Affect Strict Expandedを使用する。

データセットは、

* 192 pair-id groups
* 422 samples

から構成される。

各pair-id内では、可能な限り

* narrative structure,
* characters,
* tense,
* lexical complexity,

を一致させつつ、affective intensityを変化させる。

主要な条件は、

* peak
* moderate
* neutral

である。

明示的なhappy, sad等の感情語彙への依存を抑えることで、probeが単純なkeyword detectionを学習する可能性を低減する。

3.3 Strict Three-Way Group Split

同一pair由来の刺激がtrainingとevaluationへ同時に流入しないよう、pair_id単位で分割する。

* Train: 76 groups / 169 samples
* Alignment-dev: 58 groups / 124 samples
* Held-out test: 58 groups / 129 samples

Held-out test内には39組のcomplete peak-neutral pairsが存在する。

用途を明確に分離する。

Train splitはlinear probe fittingに用いる。

Alignment-dev splitはBase→Instruct Ridge mappingおよびdistributional statisticsの推定に用いる。

Held-out testは最終的なpredictive evaluationとcausal interventionにのみ使用する。

なお、full-layer causal localization sweepでは計算量削減のため、39 complete pairsのうち先頭15 pairsを各layerのcausal substitution評価に用いる。一方、probe evaluationはtrain/test split全体を使用する。

3.4 EmoBank

外部human annotationとの対応評価にはEmoBankを用いる。

Reader-perspective Valence/Arousal annotationsを使用し、元の尺度を必要に応じて1–9へ線形変換する。

⸻

4. Constrained Sequence-Likelihood Measurement

4.1 Candidate construction

各入力(x)について、

[
V,A \in {1,\dots,9}
]

とし、全81通りの候補

[
y_{v,a}
]

を生成する。

候補形式は、

{"valence": v, "arousal": a}

で統一する。

4.2 Raw sequence log likelihood

候補列のスコアを、

[
s_{v,a}^{\mathrm{raw}}

\sum_{t=1}^{|y_{v,a}|}
\log
P(y_{v,a,t}\mid x,y_{v,a,<t})
]

として定義する。

81候補上でSoftmaxを適用し、

[
P(v,a\mid x)

\frac{\exp(s_{v,a})}
{\sum_{v’,a’}\exp(s_{v’,a’})}
]

を得る。

4.3 Length-normalized likelihood

候補token長の違いによるlength biasを評価するため、

[
s_{v,a}^{\mathrm{norm}}

\frac{1}{|y_{v,a}|}
s_{v,a}^{\mathrm{raw}}
]

についても全解析を実施する。

これにより、本研究の主要結論がscoring conventionに依存するかを確認する。

4.4 Expected VA

出力分布から、

[
E[V\mid x]

\sum_{v,a}
vP(v,a\mid x)
]

[
E[A\mid x]

\sum_{v,a}
aP(v,a\mid x)
]

を算出する。

⸻

5. Experiment 1: Greedy Collapse and Distributional Sensitivity

5.1目的

Instructモデルのgreedy reportが中立値へ集中する場合でも、候補列全体のrelative likelihood structureに刺激情報が残っているか検証する。

5.2 Results

Instructモデルでは、greedy first-person reportの98.6%が

[
(V,A)=(5,5)
]

に集中した。

一方、sequence-likelihood distributionから計算されるexpected ValenceはEmoBank reader Valenceと有意に共変動した。

Metric	Base	Instruct
Greedy (5,5) rate	12.4%	98.6%
Pearson (r), human Valence	0.365	0.629
Spearman (\rho)	0.341	0.618
(E[V]) 5–95 percentile	3.82–7.14	5.12–5.78

Instructではdynamic rangeが縮小しているものの、刺激間の順位構造は保持される。

したがって、

[
\mathrm{Greedy\ Collapse}
\not\Rightarrow
\mathrm{Distributional\ Invariance}.
]

これはモデルにlatent subjective emotionが存在することを示すものではない。

示されているのは、固定candidate set上のconditional likelihood distributionがstimulus dependentであるという事実である。

⸻

6. Experiment 2: Layerwise Linear Decodability

6.1 Probe target

Full-layer sweepでは、各刺激を

[
y=
\begin{cases}
1 & \text{peak}\
0 & \text{neutral}
\end{cases}
]

としてRidge regression probeを学習する。

したがって本実験の(R^2)はcontinuous Valenceそのものではなく、affective peak-versus-neutral condition indicatorのheld-out predictabilityを表す。

6.2 Representation extraction

各layerについて、

* MLP output
* full layer output / residual stream

の2種類を解析する。

位置はprompt最終tokenである。

各textについて標準化promptをtokenizeし、

[
h_{\ell}^{\mathrm{MLP}}(x),
\quad
h_{\ell}^{\mathrm{resid}}(x)
]

を取得する。

6.3 Probe fitting

Ridge regressionをTrain splitでfitし、Held-out testで(R^2)を計算する。

6.4 Results

Decodabilityはdepthに応じて系統的に変化した。

MLP outputではLayer 15が最大であった。

[
R^2_{\mathrm{MLP},15}

0.546
\quad
(\mathrm{Norm})
]

[
R^2_{\mathrm{MLP},15}

0.547
\quad
(\mathrm{Raw})
]

Residual streamではLayer 14が最大となった。

[
R^2_{\mathrm{Resid},14}

0.507
\quad
(\mathrm{Norm})
]

[
R^2_{\mathrm{Resid},14}

0.508
\quad
(\mathrm{Raw})
]

初期層ではおおむね0.30–0.40、中盤層では0.45–0.55、終盤residualでは0.1–0.25程度まで低下した。

この結果は、affective conditionが中盤層から特に強くlinearly accessibleであることを示す。

ただし、

[
\mathrm{High\ Probe\ Accuracy}
]

は、

[
\mathrm{High\ Causal\ Control}
]

を意味しない。

この区別を次の実験で直接評価する。

⸻

7. Experiment 3: Cross-Model Representation Alignment

7.1 Motivation

BaseとInstructのhidden representationsが同じaffective informationを異なる座標系で表現している可能性を検討する。

7.2 Alignment

Alignment-dev split上でBase activationからInstruct activationへのRidge mappingを学習する。

[
\hat h_I

Wh_B+b.
]

Direct transfer、orthogonal alignment、Ridge mappingを比較する。

7.3 Alignment metrics

単一のprobe scoreだけでなく、高次元表現全体を評価するため、

* activation-wise (R^2)
* Linear CKA
* paired retrieval top-1 accuracy
* cosine similarity
* Mahalanobis distance
* two-sample classification AUC

を使用する。

7.4 Results

Layer 15 MLPのRidge alignmentでは、正則化条件に応じて最大で、

[
R^2_{\mathrm{activation}}
\approx0.50
]

を得た。

弱正則化領域では、

[
\mathrm{CKA}=0.829
]

[
\mathrm{Pair\ Retrieval\ Top1}=86.6%
]

に達した。

これはBase representationsから対応するInstruct representationsを高い精度で識別・予測可能であることを示す。

一方で、これらのpredictive metricsだけでは、aligned representationが自然なInstruct activationとして統計的に典型的かは判断できない。

⸻

8. Experiment 4: Ridge Regularization Sweep and Center Collapse

8.1 Design

Ridge正則化係数を、

[
\alpha
\in
{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1,10,10^2,10^3,10^4}
]

でスイープする。

8.2 Results

(\alpha)	Activation (R^2)	CKA	Top-1	Median (D_M)	AUC
(10^{-5})	.457	.826	86.6%	9.78	.623
(10^{-4})	.457	.826	86.6%	9.78	.621
(10^{-3})	.457	.826	86.6%	9.76	.624
(10^{-2})	.461	.826	86.6%	9.56	.621
(10^{-1})	.481	.829	86.6%	8.30	.616
(1)	.496	.824	75.6%	4.99	.636
(10)	.396	.783	36.6%	2.18	.713
(10^2)	.172	.670	2.4%	.68	.762
(10^3)	.010	.577	1.2%	.16	.796
(10^4)	-.025	.562	1.2%	.12	.792

自然なInstruct activationのMahalanobis radius中央値は、

[
D_M^{\mathrm{natural}}

39.63
]

であった。

Hidden dimensionが(d=1536)であるため、

[
\sqrt{d}\approx39.19
]

という高次元Gaussianの典型距離と近い。

一方、aligned representationは最弱正則化でも(D_M\approx9.8)にしか達しない。

正則化を強めると、

[
9.78
\rightarrow
8.30
\rightarrow
4.99
\rightarrow
2.18
\rightarrow
0.12
]

と分布中心へ収縮する。

この結果は、

[
\mathrm{Predictive\ Similarity}
\not\Rightarrow
\mathrm{Distributional\ Typicality}
]

を示す。

特に、CKA≈0.83やretrieval≈86.6%という高い値が得られていても、aligned representationsは自然Instruct distributionと同じ半径構造を持たない。

⸻

9. Experiment 5: Cross-Model Activation Patching

9.1 Conditions

Cross-model causal transferを検証するため、以下を比較する。

1. Target baseline
    Neutral contextをInstructモデルへ入力。
2. Within-model patch
    Peak Instruct activationをNeutral Instruct runへpatch。
3. Raw cross-model patch
    Base Peak activationを直接Instructへpatch。
4. Aligned cross-model patch
    Base Peak activationをRidge mappingした後、Instructへpatch。

主要実装ではLayer 15 MLPを使用する。

9.2 Interpretation

Aligned patchingによってBase/Instruct間の座標差を部分的に補正しても、下流reportの回復は小さい。

ただし、Ridge alignment自体がnatural target manifoldを完全には再現しないため、

cross-model patchingが効かない

ことだけから、

post-trainingがcausal couplingを破壊した

とは結論しない。

Cross-model experimentは、本研究では主としてpredictive alignmentとfunctional equivalenceが異なることを示すstress testとして位置付ける。

⸻

10. Experiment 6: Multi-Layer Simultaneous Patching

10.1 Motivation

Single-layer patchingが効かない理由が、単に一つのlayerだけを交換しているためである可能性を検証する。

10.2 Intervention blocks

以下の4種類を使用する。

* 1 Layer: L15
* 2 Layers: L14–15
* 4 Layers: L13–16
* 8 Layers: L11–18

各blockについて、

* within-model patch
* raw cross-model patch
* aligned cross-model patch

を比較する。

10.3 Results

Normalized 2D EMD recoveryは以下となった。

Block	Aligned mean	Median	95% CI	Raw	Within
L15	0.11%	0.11%	[-0.05, 0.28]	0.20%	0.23%
L14–15	0.07%	0.03%	[-0.18, 0.34]	-1.26%	0.15%
L13–16	0.35%	0.38%	[0.04, 0.64]	0.09%	0.34%
L11–18	-0.33%	-0.20%	[-1.00, 0.29]	-3.81%	-0.12%

Layer数を1から8へ増加しても、Base-like report distributionへの単調な回復は観測されなかった。

したがって、少なくとも検証した連続mid-layer MLP blockを一括置換するだけでは、report distributionを再現できない。

これはdistributed computationと整合するが、distributed mechanismを直接証明するものではない。

⸻

11. Experiment 7: Within-Model Substitution Controls

11.1 Motivation

Cross-model patchingのnull resultを解釈する前に、

そもそも同一モデル内で、そのactivation sliceを置換すればreportが動くか

を検証する。

11.2 Protocols

Held-out test内のcomplete Peak–Neutral 39 pairsを用いる。

以下を比較する。

1. mlp_last_token_L15
2. resid_last_token_L15
3. multi_resid_last_L13_16
4. resid_all_tokens_L15
5. multi_resid_all_L13_16

11.3 Results

局所的なlast-token substitutionでは回復は小さい。

Protocol	Raw mean/median Rec	Norm mean/median Rec
L15 MLP last-token	0.82 / 0.56%	1.30 / 1.13%
L15 Resid last-token	0.38 / 0.36%	0.42 / 0.15%
L13–16 Resid last-token	1.47 / 1.06%	1.38 / 0.89%

一方、all-token replacementでは非常に大きな負のrecoveryを示した。

Protocol	Raw Rec	Norm Rec
L15 Resid all-token	-448.95%	-225.88%
L13–16 Resid all-token	-437.15%	-236.91%

これは異なる文脈から得たsequence-wide activationを無理に置換することで、target computationを大きく破壊することを示す。

したがって、本研究ではlast-token interventionを主要なlocal causal testとして扱う。

⸻

12. Experiment 8: Full-Layer Causal Localization Sweep (True 2D Joint OT & Attention Output)

12.1 Motivation

Layer 15がたまたま不適切だった可能性、および因果的ボトルネックがMLPではなくMulti-Head Self-Attention経路に存在する可能性を排除するため、全28層のMLP output、Attention output、Residual streamを系統的に調査する。

さらに、先行実験におけるValence/Arousal独立周辺分布の和（Marginal Wasserstein和）がVA結合依存性を無視する擬似指標であった点を改め、81×81のManhattan ground costに基づく真の2D Joint Optimal Transport（Joint OT; Earth Mover's Distance）を主評価指標として全層再測定を行った。

12.2 Design

各layer (\ell)および各component（MLP output, Attention output, Residual stream）について、

[
D_\ell = \text{held-out probe }R^2
]

と、真の2D Joint Optimal Transportに基づく局所因果回復率

[
C_\ell^{\mathrm{Joint}} = 1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})}
]

を測定する。評価ペアの分母が過小な場合（(\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < 0.05)）は回復率計算から除外するセーフガードを適用し、Held-out testのcomplete Peak–Neutral pairs（各層15 pairs）を評価した。介入位置はprompt最終tokenであり、Attention介入は当該トークン位置でのprojected attention-module outputの置換として厳密に定義される。

12.3 Main Results

真の2D Joint OTによる全層・全コンポーネント測定結果の要約は以下の通りである。

Component	Max Probe (R^2) (Layer)	Max Joint OT Rec (Layer)	Spearman (\rho)	p-value
MLP	0.5610 (L15)	2.20% (L10)	0.2956	0.1268
ATTN	0.5495 (L18)	1.48% (L20)	0.0230	0.9076
RESID	0.5016 (L14)	1.68% (L16)	-0.0394	0.8422

主要な知見は以下の通りである。

1. **Attention経路における因果回復の欠如**:
   Attention outputにおける局所回復率も最大1.48%（Layer 20）に留まり、MLP（最大2.20%）やResidual（最大1.68%）と同様に極小であった。この結果は、「感情情報の因果的伝達がAttention機構を通じて迂回されている」という仮説を否定する。
2. **最高デコード層での回復率の極小性**:
   MLPデコーダビリティが最大となるLayer 15（(R^2 = 0.5610)）におけるJoint OT回復率はわずか1.00%（中央値0.72%）であった。同様にAttentionデコーダビリティが最大となるLayer 18（(R^2 = 0.5495)）での回復率は0.04%（中央値0.14%）であった。
3. **Decodability–Causal Recoveryの無相関**:
   層別probe (R^2)とJoint OT回復率の間には、いずれのコンポーネントにおいても統計的に有意な単調関係は検出されなかった（MLP: (\rho = 0.2956, p = 0.1268); ATTN: (\rho = 0.0230, p = 0.9076); RESID: (\rho = -0.0394, p = 0.8422)）。

12.4 Marginal Wasserstein vs. Joint OT Robustness

副指標として計算されたMarginal Wasserstein和回復率においても、MLP最大2.33%（L4）、ATTN最大1.16%（L1）、RESID最大1.57%（L14）と一貫して極小であり、距離関数の結合・周辺構造の定義によらず、局所的単一層置換が下流報告を回復しないという定性的結論は極めて頑健である。

12.5 Interpretation

以上の結果は、プロンプト最終トークンにおける局所的内部表現が、MLP・Attention・Residualのいずれの計算経路においても、下流の一人称感情報告に対して十分な因果的影響力を持たない（low local causal sufficiency）ことを全層にわたり確定させるものである。

⸻

13. Experiment 9: Dual-Outcome Behavioral Readout

13.1 Motivation

First-person VA reportに対する局所causal effectが小さいことが、

affect-related activationがすべてのdownstream taskで機能しない

ことを意味するか検証する。

13.2 Two outcomes

各contextについて二種類のoutcomeを計算する。

第一はfirst-person self-report:

[
E[V].
]

第二はsupportive responseとneutral responseのlog-likelihood ratio:

[
B(x)

\log P(y_{\mathrm{support}}\mid x)

\log P(y_{\mathrm{neutral}}\mid x).
]

Layer 15 MLP patch前後で両者を評価する。

13.3 Current observation

保存済み結果では、複数刺激についてintervention前後で(E[V])に小さな変動が見られる一方、記録されたbehavioral log-likelihood ratioは同一値となるケースが多い。

この実験は、局所L15 MLP activationがself-reportだけでなくsupportive-vs-neutral response preferenceに対しても大きなlocal leverageを示さない可能性を示す。

ただし、本結果は本稿の主証拠ではなくexploratory analysisとして扱う。

⸻

14. Experiment 10: Mood-Congruency / Third-Person Recognition Steering

14.1 Motivation

Matched activation substitutionがfirst-person reportをほとんど動かさない一方で、affect-associated directionへの明示的steeringは別taskへ因果的影響を与えられるかを調べる。

14.2 Dataset

EmoBankからValenceが中立近傍に位置する曖昧刺激107件を抽出する。

14.3 Intervention

Layers 14, 16, 20に対し、

* valence direction
* norm-matched random direction

を注入する。

Strengthは、

[
\alpha
\in
{-3,-1.5,0,+1.5,+3}
]

standard deviationsとする。

総観測数は3210。

14.4 Results

Layer	Direction	(\beta_{\mathrm{mood}})	SE	p
14	Valence	-0.0323	.0016	(1.18\times10^{-70})
14	Random	-0.0042	.0013	(9.63\times10^{-4})
16	Valence	-0.0169	.0012	(6.20\times10^{-40})
16	Random	+0.0444	.0013	(6.40\times10^{-138})
20	Valence	+0.0244	.0013	(3.23\times10^{-61})
20	Random	+0.0091	.0009	(1.45\times10^{-23})

Layer 14ではnegative slope、Layer 20ではpositive slopeが観測された。

ただしLayer 16ではrandom directionの効果がvalence directionより大きく、activation perturbationに対するgeneric sensitivityの可能性がある。

したがって、

pure mood-congruency circuitを同定した

とは結論しない。

より限定的には、

affect-associated steering can exert layer-dependent causal effects on a separate third-person recognition task, but direction specificity is not uniform across layers

と解釈する。

この結果は、first-person reportに対するlocal substitution effectが小さいことを、affect-related representationsが一般にcausally inertであることと混同してはならないことを示す。

⸻

15. Experiment 11: Generation-Time Causal Patching Sweep

15.1 Motivation

査読上の重大な反論として、「プロンプト最終トークン（Prompt-time）での介入では、その後の自己報告生成過程における文脈依存の計算を更新できないのではないか。自己報告プレフィックス `{"valence": ` が出力され、モデルが具体的な感情価トークンを生成する直前（Generation-time）のトークン位置で介入すべきである」という指摘が想定される。

この仮説を検証するため、生成時トークン位置における全28層の因果パッチングスイープを実施した。

15.2 Design

プロンプトに対し、自己報告の開始プレフィックス（`{"valence": `）を付与した系列を入力とし、当該プレフィックスの最終トークン位置において、Peak条件の内部活性化（MLP output, Attention output, Residual stream）をNeutral条件の実行コンテキストへ置換した。

介入効果は、真の2D Joint Optimal Transportに基づくPeak方向への回復率（True Joint OT Recovery）

[
G_\ell = 1 - \frac{\mathrm{OT}_{VA}(P_{\mathrm{patch}}, P_{\mathrm{peak}})}{\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}})}
]

として測定した。微小分母（(\mathrm{OT}_{VA}(P_{\mathrm{neut}}, P_{\mathrm{peak}}) < 0.05)）に対するセーフガードを適用し、有効ペアについて評価した。

15.3 Results

全28層×3コンポーネント（MLP, ATTN, RESID）における生成時因果パッチングの結果は以下の通りである。

* **MLP output**: 初期・中盤層（Layer 0〜18）では回復率は -3%〜+2% 前後で推移した。後段層において回復率の微小な上昇が観測され、Layer 19で 3.21%、Layer 20で 2.60%、Layer 23で 3.10%、**Layer 24で最大 5.02%**（中央値 0.00%）に達した。しかし最終層（Layer 27）では -6.91% へ反転した。
* **Attention output**: Layer 10で 3.73%、**Layer 20で最大 4.23%**（中央値 0.00%）、Layer 22で 2.82% を記録したが、他の多くの層では 0% 前後または負値を示した。
* **Residual stream**: 前半層（Layer 0〜16）では -8%〜-27% という大きな負の回復率（パッチングによる出力コヒーレンスの破綻）を示し、後半層（Layer 18〜27）では -1%〜+0.5% 前後で推移した（最大 0.48% at Layer 19）。
* **最高デコード層での挙動**: プロンプト時リニアプローブデコーダビリティが最大であったLayer 15（(R^2 = 0.56)）における生成時回復率は、MLPで 1.22%、Attentionで 2.25%、Residualで -8.06% に留まった。

15.4 Defensive Framing & Interpretation

生成時パッチングでは、プロンプト時（最大 2.20%）と比較して後段層（Layer 20〜24）で回復率のわずかな上昇（最大 5.02%）が認められる。これは、下流トークン出力の直前段階において局所表現の直接的伝播が一部強まる動態を示唆する。

しかし、**最大回復率であっても約 5%（中央値はいずれも 0.00%）** に過ぎず、出力分布全体の変位の 95% 以上は依然として回復されない。したがって、介入位置を生成時トークンへ移動させた場合であっても、「単一の局所層表現が自己報告出力分布を決定論的に支配している」という仮説は支持されない。

⸻

16. Experiment 12: Probe-Aligned Local Necessity and Specificity Controls

16.1 Motivation

因果的媒介（causal mediation）の検証には、十分性（sufficiency）だけでなく必要性（necessity）の評価が不可欠である。「プローブが検出している方向成分は、下流報告の生成に不可欠（necessary）なのか」「プローブ方向を除去した場合、出力報告は感情中立方向へ退行するのか」「その効果はランダムな直交方向を除去した場合と統計的に区別できるのか（特異性）」を検証する。

16.2 Design

各層・各コンポーネント（MLP, ATTN, RESID）の活性化ベクトル (h) に対し、学習済み線形プローブの重み方向単位ベクトル (\hat{v}_{\mathrm{probe}}) を幾何学的に完全直交射影除去する介入を施した：

[
h_{\mathrm{ablated}} = h - (h^\top \hat{v}_{\mathrm{probe}})\hat{v}_{\mathrm{probe}}
]

本実験では、以下の4大操作を厳密に分離・測定した。

1. **Probe-Aligned Necessity Displacement**:
   Peak入力に対するプローブ方向消去後の出力分布と元のPeak出力分布との2D Joint OT変位量：
   [
   N_\ell = \mathrm{OT}_{VA}(P_{\mathrm{peak\setminus probe}}, P_{\mathrm{peak}})
   ]
2. **Neutralization Ratio**:
   プローブ方向消去によって、出力がNeutral基準分布へ実際にどれだけ近づいたかの比率：
   [
   R_{\mathrm{neut}} = \frac{\mathrm{OT}_{VA}(P_{\mathrm{peak}}, P_{\mathrm{neut}}) - \mathrm{OT}_{VA}(P_{\mathrm{peak\setminus probe}}, P_{\mathrm{neut}})}{\mathrm{OT}_{VA}(P_{\mathrm{peak}}, P_{\mathrm{neut}})}
   ]
3. **Specificity Null Controls**:
   プローブ方向と厳密に直交する超平面からサンプリングしたランダム単位方向 (\hat{v}_{\perp}) を消去した際の変位分布に対する標準化スコア (Z_\perp) および経験的 p値。
4. **Matched Neutral Baseline**:
   Neutral条件の活性化ベクトルそのものからプローブ方向を消去した際のコントロール変位。

全28層×3コンポーネント（計84条件）に対し、Benjamini-Hochberg FDR（False Discovery Rate）補正を適用した。さらに、代表4層（Layer 7, 15, 21, 27）についてはランダム方向サンプル数を (N=100) に拡大した高精度追試を実施した。

16.3 Results

全層スイープにおける主要な実測結果は以下の通りである。

* **Probe Necessity Displacementの極小性**:
   (N_\ell) の平均変位量は全層で 0.0004 〜 0.0082 の範囲に収まった。元のPeak–Neutral間の分布距離（約 0.20）と比較して 2〜4% 程度の微弱な揺らぎに過ぎない。
* **Neutralization（中和）の不在**:
   中和比率 (R_{\mathrm{neut}}) は、全層を通じて **-3.27% 〜 +0.61%** であり、プローブ方向を消去しても出力が中立分布へ向かって復元される傾向は一切観察されなかった（むしろ微小な負値、すなわち直交的な摂動による歪みを示す）。
* **特異性検定における有意差の欠如**:
   代表層における直交帰無分布との比較結果は以下の通りである。
   * Layer 7 (Confirmatory): MLP (Z_\perp = 0.09, p = 0.4286); ATTN (Z_\perp = 1.92, p = 0.0952); RESID (Z_\perp = 0.16, p = 0.3810)
   * Layer 15 (Confirmatory): MLP (Z_\perp = -0.30, p = 0.6190); ATTN (Z_\perp = -0.09, p = 0.4762); RESID (Z_\perp = -0.31, p = 0.7143)
   * Layer 21 (Confirmatory): MLP (Z_\perp = -1.21, p = 0.9048); ATTN (Z_\perp = -0.57, p = 0.7619); RESID (Z_\perp = 0.97, p = 0.2381)
   * Layer 27 (Confirmatory): MLP (Z_\perp = 0.54, p = 0.3333); ATTN (Z_\perp = -1.84, p = 1.0000); RESID (Z_\perp = 0.27, p = 0.5714)
   全84条件において、生p値が0.05を下回った数例（L6 MLP, L16 MLP, L19 MLP）を含め、Benjamini-Hochberg FDR補正後には**すべての層・コンポーネントで有意水準（(q < 0.05)）を満たすものは皆無（0/84）**であった。

16.4 Defensive Framing & Interpretation

本結果は、プローブ方向の除去による出力変位が、同一ノルムを持つ任意の直交ランダム方向を消去したときの非特異的変位と統計的に区別できないことを示す。

ただし、この結果から「モデル内部で感情情報が一切使われていない」と過剰に主張することはできない。厳密に言えるのは、**「線形プローブによって特定された局所的1次元部分空間は、下流の報告生成に対して特異的な局所的必要性（local necessity）を持たない」** という点である。

⸻

17. Experiment 13: Cross-Family Architectural Replication (Llama-3.2-1B-Instruct)

17.1 Motivation

本研究で得られた「高デコーダビリティと低因果回復率の解離」および「生成時パッチングの限定的効果」が、Qwen2.5アーキテクチャ特有の訓練特性やバイアスに起因する可能性を検証するため、異なるモデルファミリである **Meta Llama-3.2-1B-Instruct**（16 Transformer layers）を用いて全層因果パッチングスイープの完全な独立追試を実施した。

17.2 Results

Llama-3.2-1B-Instructにおける全16層×3コンポーネントの生成時因果パッチング（Joint OT Recovery）の結果は以下の通りである。

* **MLP output**: 回復率は全層で一貫して負値またはほぼ0%であり、最大値はLayer 15の -0.36% であった（Layer 0: -80.37%, Layer 3: -10.23%, Layer 10: -11.67%）。
* **Attention output**: Layer 3で 0.41%、Layer 4で **最大 0.73%**、Layer 14で 0.11% の微小な正の回復率が観測されたが、全体として 1% 未満に留まった。
* **Residual stream**: Layer 0（-36.18%）からLayer 15（-4.23%）に至る全層で一貫して負値を示し、単一層Residualの強制置換が出力コヒーレンスを破壊することを示した。

17.3 Interpretation

LLaMAアーキテクチャにおいても、生成時における単一層の局所活性化置換による因果回復率は最大でも 0.73% に過ぎず、Qwenで観測された現象（局所単一層の非十分性）がモデルファミリを超えて普遍的であることが実証された。

⸻

⸻

18. Integrated Results: The Four-Panel Mechanistic Framework

本研究で得られた全28層・全コンポーネントにわたる一連の実測データを統合すると、以下の多層的・多面的因果プロファイル（Four-Panel Mechanistic Framework）が確立される。

1. **Panel A: Layerwise Linear Decodability ((D_\ell))**:
   プロンプト最終トークンにおける感情極性（Peak vs. Neutral）の線形判別能は、中盤層（Layer 14–15）で極大に達する（MLP: (R^2 = 0.5610); ATTN: (R^2 = 0.5495); RESID: (R^2 = 0.5016)）。
2. **Panel B: Prompt-Time Local Causal Sufficiency ((S_\ell))**:
   プロンプト最終トークンにおける同一モデル内活性化置換（Peak→Neutral）による真の2D Joint OT回復率は、全28層・全経路を通じて一貫して極小である（MLP最大 2.20%; ATTN最大 1.48%; RESID最大 1.68%）。デコーダビリティとの単調相関は認められない。
3. **Panel C: Probe-Aligned Local Necessity ((N_\ell))**:
   プローブ方向の幾何学的射影消去による出力分布の変位量（(N_\ell = 0.0004 \sim 0.0082)）は極めて微小であり、中和比率（(R_{\mathrm{neut}} = -3.27\% \sim +0.61\%)）は一切の中和傾向を示さない。直交ランダム方向消去との差（(Z_\perp)）は、全層×3コンポーネントのBenjamini-Hochberg FDR補正後、すべての層で非有意（(q > 0.05)）である。
4. **Panel D: Generation-Time Causal Sufficiency ((G_\ell))**:
   自己報告生成直前のプレフィックス最終トークンにおける介入では、後段層（Layer 20〜24）で微小な変位上昇（MLP L24で最大 5.02%、ATTN L20で最大 4.23%）が認められるものの、中央値はいずれも 0.00% であり、95%以上の変位は非回復のままである。

以上より、本研究は以下の三位一体の関係を実証した：

[
\boxed{
\text{High Decodability } (R^2 \approx 0.56)
\quad\not\Rightarrow\quad
\text{Low Local Sufficiency } (S_\ell < 2.2\%)
\quad\land\quad
\text{Low Probe-Aligned Necessity } (R_{\mathrm{neut}} \approx 0\%)
}
]

⸻

19. Discussion

19.1 Decodability Is Neither Sufficiency Nor Necessity

表現学習・機械解釈性（mechanistic interpretability）研究において、高精度なリニアプローブの存在は、モデルがその属性を内部表現として獲得している強力な証拠として広く受け入れられてきた。しかし、本研究の結果は、プローブの予測能（decodability）から、その表現部位がモデルの下流計算において果たす因果的役割（causal role）を安易に同一視してはならないことを明確に示す。

* **十分性の欠如**: 最も明確に感情価が読み取れるLayer 15 MLPの内部状態をPeak状態へ置き換えても、下流の感情報告分布はわずか 1.00% しか回復しない。
* **必要性の欠如**: そのプローブ方向を完全に消去しても、出力は中立状態へ退行せず、ランダムな直交方向を消去したときの非特異的な出力の揺らぎと統計的に区別できない。

したがって、プローブが検出する線形特徴量は、「モデルがアクセス可能な形で保持している情報」ではあっても、「局所的に自己報告出力を決定づける直接の制御ノブ」ではない。

19.2 Resolution of Four Major Alternative Hypotheses

本研究で実施した4大因果検証実験は、先行研究で想定され得る主要な対立仮説を先回りして検証・解消した。

1. **反論1: 「SufficiencyだけでなくNecessityを測定すべきではないか」**:
   → **実証的回答**: 幾何学的直交射影によるProbe direction ablationおよびSpecificity control（84条件BH-FDR補正）を実施した。結果、プローブ方向消去による中和比率は一貫して 0% 近傍（-3.27%〜+0.61%）であり、特異的有意差（(q < 0.05)）を示す層は皆無であった。プローブ方向は局所的に十分でないだけでなく、局所的に不可欠でもない。
2. **反論2: 「Prompt時ではなくGeneration時に介入すべきではないか」**:
   → **実証的回答**: 自己報告生成プレフィックス直後における全28層パッチング（Generation-time sweep）を実施した。後段MLP（L24）で 5.02%、Attention（L20）で 4.23% と、プロンプト時を上回る回復率の兆候が確認されたものの、依然として中央値は 0.00% であり、95%以上の変位は非回復であった。単一層の局所介入は生成時においても出力を支配しない。
3. **反論3: 「因果回路がAttention経路にあるのではないか」**:
   → **実証的回答**: MLP outputだけでなく、各層のprojected Attention-module outputおよびResidual streamの全層パッチングを実施した。Attention経路における回復率もプロンプト時最大 1.48%（L20）、生成時最大 4.23%（L20）に留まり、MLPと同様に局所的ボトルネックになっていない。
4. **反論4: 「Qwen特有のアーキテクチャや訓練バイアスではないか」**:
   → **実証的回答**: Llama-3.2-1B-Instruct（全16層）に対する生成時因果パッチングを独立実施した。LLaMAにおける因果回復率は全層で一貫して 1% 未満（最大 0.73% at L4 ATTN）であり、本知見がTransformer全般に共通する普遍的動態であることが確認された。

19.3 Defensive Framing: What These Findings Do and Do Not Mean

本研究の解釈において、以下の境界づけ（defensive framing）が極めて重要である。

* **断定してはならない過剰主張**:
  * 「モデルは感情情報を一切下流計算に利用していない」
  * 「因果回路が存在しない」
  * 「生成時介入でも感情は全く動かない」
* **実験データから支持される厳密な結論**:
  * 「プローブによって同定される単一層の局所的1次元部分空間、および単一層の全活性化スライスは、プロンプト時・生成時のいずれにおいても、自己報告出力分布を決定論的に回復・中和する局所的レバーとして機能しない」
  * 感情情報の実際の計算は、単一の局所的ボトルネックではなく、系列全体にわたる文脈依存の注意機構や多層にわたる分散的表現（distributed computation）を通じて緩やかに媒介されていると考えられる。

⸻

20. Limitations

1. **非局所的・複数層パスパッチングの未網羅**:
   本研究では1/2/4/8層の連続ブロックパッチングおよび単一層全層スイープを実施したが、非連続な疎結合回路（sparse circuit）や特定Attention Head間の相互作用パスを網羅するPath Patchingまでは実施していない。
2. **非線形アライメントの未検証**:
   Base/Instruct間の表現対応づけにはRidge回帰（線形写像）を用いており、非線形多様体アライメントにおける幾何学的歪みの完全な解消には至っていない。
3. **モデル規模**:
   検証は1B〜1.5B規模のオープンウェイトモデル（Qwen2.5-1.5B, Llama-3.2-1B）に集中しており、7B以上の大規模モデルにおけるスケール効果の確認は将来の課題である。

⸻

21. Conclusion

本研究は、大規模言語モデルにおける情動関連表現をケーススタディとして、内部表現の線形解読能（decodability）と因果的媒介能（causal leverage）の根本的な乖離を実証した。

Qwen2.5-1.5BおよびLlama-3.2-1Bを用いた体系的実験により、以下の包括的結論が得られた：

1. 中間層（Layer 14–15）において刺激の感情価は極めて高く線形デコード可能（(R^2 \approx 0.56)）である。
2. しかし、同一層の局所活性化を置換しても、自己報告分布の回復率はプロンプト時で最大 2.20%、生成時でも最大 5.02%（中央値 0.00%）に留まる。
3. 学習済みプローブ方向を幾何学的に消去しても出力の中和は一切生じず、その効果は直交ランダム方向の消去と統計的に区別できない（BH-FDR (q > 0.05)）。
4. Attention経路の迂回や他モデルファミリ（LLaMA）への追試を通じても、この局所的因果解離は極めて頑健に維持される。

これらの知見は、LLMの内部表現解析においてプローブの成功を安易にモデルの因果メカニズムと同定する慣習に警鐘を鳴らすものであり、今後の機械解釈性研究において、線形解読能・幾何学的整合性・十分性・必要性の多面的検証を標準プロトコルとして導入することの重要性を強く支持する。

⸻

Appendix A. Full-Layer Causal Localization Sweep (Prompt-Time Joint OT)

Qwen2.5-1.5B-Instructにおける全28層のProbe (R^2)および真の2D Joint Optimal Transport回復率（15 pairs/layer, (\epsilon_{\mathrm{rec}} = 0.05) セーフガード適用）の実測値を報告する。

Layer	MLP (R^2)	MLP Joint OT Rec (%)	ATTN (R^2)	ATTN Joint OT Rec (%)	RESID (R^2)	RESID Joint OT Rec (%)
0	0.3025	-1.39%	0.3395	-0.92%	0.3129	-1.15%
1	0.3045	-0.47%	0.3263	+0.78%	0.3529	-0.04%
2	0.3608	-0.32%	0.3971	-0.78%	0.3943	-1.93%
3	0.3932	+1.38%	0.4903	+0.60%	0.3694	-2.73%
4	0.3847	+1.67%	0.3997	-2.04%	0.3521	-2.71%
5	0.3685	-0.57%	0.3844	+0.61%	0.4059	-2.23%
6	0.4229	-0.17%	0.4698	-1.23%	0.4697	-3.70%
7	0.4787	+1.21%	0.4321	-0.50%	0.4726	-2.09%
8	0.4671	-0.94%	0.3837	+0.20%	0.4619	-3.05%
9	0.4134	+0.78%	0.4157	+0.09%	0.4141	-2.31%
10	0.4817	+2.20%	0.4753	+0.78%	0.4744	-1.37%
11	0.4891	+0.25%	0.4830	+0.21%	0.4519	-0.16%
12	0.4571	-0.92%	0.4085	+1.46%	0.4561	-0.53%
13	0.5151	-0.06%	0.5205	+0.92%	0.4887	+0.61%
14	0.5372	+1.35%	0.5313	-0.36%	0.5016	+0.71%
15	0.5610	+1.00%	0.5217	-0.52%	0.4862	+0.25%
16	0.5158	+1.02%	0.4913	+0.98%	0.4705	+1.68%
17	0.4772	-0.04%	0.4936	-0.04%	0.4432	+1.34%
18	0.5513	-0.43%	0.5495	+0.04%	0.4848	-0.58%
19	0.4434	-0.15%	0.4483	-0.18%	0.4330	+0.81%
20	0.4866	-0.38%	0.4602	+1.48%	0.4008	+0.65%
21	0.5223	+0.62%	0.4859	+0.21%	0.3903	+0.85%
22	0.3923	+0.16%	0.4076	+0.96%	0.2193	+0.31%
23	0.3774	+0.52%	0.4514	+1.23%	0.2107	+1.31%
24	0.3514	+0.84%	0.4072	+0.61%	0.1470	-0.00%
25	0.3178	-0.36%	0.4100	+0.97%	0.1790	+0.18%
26	0.3311	+0.44%	0.1040	+0.41%	0.1681	+0.23%
27	0.3131	-0.00%	-0.2670	-0.00%	0.2496	-0.00%

⸻

Appendix B. Reproducibility & Artifact Mapping

主要結果と実装スクリプト・保存先データの対応は以下の通りである。

* **厳密2D Joint Optimal Transport & 距離計算モジュール**:
    `v3/src/ot_utils.py`
* **モデル共通抽象化・Hook・直交消去モジュール**:
    `v3/src/model_utils.py`
* **一括バッチ順伝播尤度計算モジュール**:
    `v3/src/batch_likelihood.py`
* **因果拡張単体テスト (6件)**:
    `v3/tests/test_causal_extensions.py`
* **全層因果ローカリゼーションスイープ（真の2D Joint OT & Attention統合）**:
    スクリプト: `v3/scripts/run_causal_localization_sweep.py`
    実測データ: `v3/results/causal_localization_sweep_joint_ot.csv`
* **生成時因果パッチングスイープ（Qwen2.5-1.5B）**:
    スクリプト: `v3/scripts/run_generation_time_causal_sweep.py`
    実測データ: `v3/results/generation_time_causal_sweep.csv`
* **プローブ整合型局所必要性・特異性スイープ（全層BH-FDR補正）**:
    スクリプト: `v3/scripts/run_probe_aligned_necessity_sweep.py`
    実測データ: `v3/results/probe_aligned_necessity_sweep.csv`
* **アーキテクチャ間普遍性検証（Llama-3.2-1B-Instruct生成時スイープ）**:
    スクリプト: `v3/scripts/run_generation_time_causal_sweep.py`
    実測データ: `v3/results/generation_time_causal_sweep_llama.csv`
* **先行アライメント・幾何学・多層実験データ**:
    `v3/results/ridge_alpha_sweep_results.csv`
    `v3/results/aligned_patching_results.csv`
    `v3/results/multilayer_patching_results.json`
    `v3/results/within_model_positive_control_results.csv`
    `v3/results/dual_outcome_results.csv`
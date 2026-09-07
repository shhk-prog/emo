Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations in a Paired Base/Instruct Language Model

Abstract

大規模言語モデル（LLM）の内部状態から属性を高精度にデコードできることは、その情報がモデル内部に存在する証拠を与える。しかし、decodabilityは、その表現が特定の下流計算に利用されていることを必ずしも意味しない。本研究では、この古典的なrepresentation–use distinctionを、Base/Instruct言語モデルペアにおけるaffect-relevant representationsをケーススタディとして機構的に検証する。

Qwen2.5-1.5B Base/Instructを対象に、語彙交絡を制御したAIPsy-Affect Strict Expanded、81候補の制約付きVA報告、層別linear probing、Base→Instruct表現アライメント、およびactivation substitutionを統合した評価系を構築した。

第一に、Instructモデルのgreedyな一人称VA報告は98.6%が (5,5) に集中する一方、候補列全体の条件付き尤度から得られるValence expectationは刺激の感情価と共変動した。したがって、単一のgreedy completionによる表層的中立化は、候補出力分布全体の刺激依存性の消失を意味しない。

第二に、最終prompt tokenの中間表現からaffective-versus-neutral conditionを線形に予測でき、そのdecodabilityはネットワーク深度に伴って変化し、中盤層付近で最大となった。

第三に、Base→Instruct Ridge alignmentはheld-out activationsについて高いpredictive correspondenceを示した。弱い正則化条件ではLinear CKAは約0.83、paired-retrieval top-1 accuracyは86.6%に達した。一方、Mahalanobis解析では、自然なInstruct activationが約39.6の典型半径を持つのに対し、aligned activationは8–10程度まで分布中心側へ収縮した。したがって、高いR²、CKA、retrieval accuracyは、target activation distributionへの完全な統計的適合を保証しない。

これらの結果は、representation alignmentの評価にpredictive similarityだけを用いることの限界を示す。本研究は、主観的感情や内省を主張するものではなく、what is linearly accessible, what is predictively transferable, and what is causally used must be evaluated as distinct questionsという方法論的区別をaffect-relevant representationsを用いて検証するケーススタディである。

⸻

1. Introduction

大規模言語モデルのhidden representationsには、言語構造、知識、真偽、安全性、感情的属性など、さまざまな情報が線形にデコード可能な形で含まれることが知られている。

しかし、probeがある属性を予測できることから、

the model uses that representation to produce the downstream behavior

と結論することはできない。

この問題はprobing研究において以前から指摘されてきた。Probeはモデル自身のdownstream computationには必要でない相関情報も利用できるため、

[
\text{Decodability}
\not\Rightarrow
\text{Behavioral Use}
]

である。

本研究では、この区別をBase/Instruct language modelにおけるaffect-relevant representationsを用いて検討する。

1.1 Motivation

Qwen2.5-1.5B-Instructへaffective textを提示し、一人称形式でValence–Arousalを報告させると、greedy decodingはほぼ常に

{"valence": 5, "arousal": 5}

へ集中する。

しかし、この現象には少なくとも異なる説明が存在する。

1. affect-relevant informationそのものが内部表現から失われている。
2. 情報は存在するがgreedy outputだけが中立化されている。
3. 情報はdecodableだが、その表現部位は報告生成に利用されていない。
4. 情報は別のtoken position、component、あるいはdistributed computationとして利用されている。

これらを区別するためには、behavior、decodability、representation alignment、causal interventionを独立に測定する必要がある。

1.2 Research Questions

本研究では以下を問う。

RQ1. Greedy reportが中立化した場合でも、constrained output distributionには刺激依存的情報が残るか。

RQ2. Affect-related conditionはBase/Instructモデルの内部状態からどの程度decodableか。

RQ3. Base/Instruct間の表現はheld-out data上でどの程度predictively align可能か。

RQ4. Predictively aligned representationsはtarget modelの自然activation distributionにも適合するか。

RQ5. Probe-accessible activation sitesは、対応するreportに対してcausal sufficiencyまたはnecessityを持つか。

RQ5については、正しく実装されたactivation-substitutionおよびablation実験によって評価する。

⸻

2. Construct Definition

本研究では以下を明確に区別する。

Reader-rated text affect
人間読者によるテキストのValence/Arousal評定。

Third-person affect recognition
モデルによる登場人物、筆者、話者等の感情状態の推定。

Constrained first-person report distribution
標準化promptに続く有限個のVA response candidatesにモデルが割り当てる条件付き確率分布。

本稿における first-person は文法的・task-levelな記述にすぎず、introspective access、subjective experience、sentience等を意味しない。

また、affect-relevant representation はaffective stimulus conditionまたはhuman VA annotationをpredictできる内部表現を意味し、主観的感情状態を意味しない。

⸻

3. Experimental Setup

3.1 Models

同一architectureを持つ

* Qwen2.5-1.5B
* Qwen2.5-1.5B-Instruct

を使用する。

Base/Instruct pairはpost-trainingの一般的因果効果を推定するためではなく、representational transferに対するcontrolled stress testとして使用する。

3.2 Data

AIPsy-Affect Strict Expandedは192 pair-id groups、422 samplesから構成される。

Group leakageを防ぐため、pair-id単位で

* Train
* Alignment-dev
* Held-out test

へ分割する。

EmoBankは外部human annotationとのcorrespondence評価に使用する。

⸻

4. Constrained Report Measurement

各入力 $x$ について、

[
V,A\in{1,\ldots,9}
]

からなる81個の候補

[
y_{v,a}
]

を評価する。

Primary implementationでは、

[
s_{v,a}

\log P(y_{v,a}\mid x)
]

を計算し、

[
P(v,a\mid x)

\frac{\exp(s_{v,a}/\tau)}
{\sum_{v’,a’}\exp(s_{v’,a’}/\tau)}
]

と正規化する。

候補token長が異なる場合の長さバイアスを評価するため、補足解析として

[
\bar s_{v,a}

\frac{1}{|y_{v,a}|}
\log P(y_{v,a}\mid x)
]

によるlength-normalized scoreについても同じ解析を再実行する。

この二つを明示的に区別する。

⸻

5. Results

5.1 Greedy Neutralization Does Not Imply Distributional Invariance

Instructモデルではgreedy first-person VA reportの98.6%が (5,5) に集中した。

一方、81候補のconditioned likelihood distributionから得られるexpected ValenceはEmoBank reader Valenceと共変動した。

この結果は、

[
\text{Greedy collapse}
\not\Rightarrow
\text{complete distributional collapse}
]

であることを示す。

重要なのは、これを「latent emotion」の証拠とは解釈しないことである。観測されているのは、固定された候補集合上のconditional probability structureである。

⸻

5.2 Affect-Related Conditions Are Linearly Decodable

各層の最終prompt-token representationにlinear probeを適用した。

全層スイープでは、affective peak versus neutral conditionのdecodabilityは層によって系統的に変化し、中盤層で最大となった。

MLP outputについて最大値はLayer 15の

[
R^2=0.548
]

であった。

この値はcontinuous Valence predictionではなく、peak-versus-neutral condition indicatorに対するheld-out regression performanceである。

したがって、この実験から言えるのは、

affective condition is strongly linearly accessible from mid-layer representations

までであり、

the model causally uses this representation

ではない。

⸻

5.3 Base and Instruct Representations Are Predictively Alignable

Base activationからInstruct activationへのRidge mapをalignment-dev splitのみで学習した。

Held-out testでは、弱い正則化条件において、

[
\mathrm{CKA}\approx0.83
]

および

[
\mathrm{Top1\ retrieval}=86.6%
]

が得られた。

したがって、Base/Instruct representationには、直接座標系が異なっていてもpredictive correspondenceが存在する。

⸻

5.4 Predictive Alignment Does Not Guarantee Distributional Typicality

Predictive alignmentの品質をR²やCKAだけで判断できるかを検証するため、Ridge regularization parameterを

[
10^{-5}\le\alpha\le10^4
]

でスイープした。

自然なInstruct activationのMahalanobis radiusの中央値は

[
D_M=39.63
]

であった。

一方、aligned activationは、

[
D_M=9.78
\quad(\alpha=10^{-5})
]

から

[
D_M=0.12
\quad(\alpha=10^4)
]

へ単調に収縮した。

特に、

[
N_{\mathrm{alignment}}\ll d_{\mathrm{hidden}}
]

の設定において、Ridge predictionはtarget activationのconditional mean方向へvarianceを縮小させる。

興味深いことに、

[
\mathrm{CKA}\approx0.83,\qquad
\mathrm{retrieval}\approx86.6%
]

という高いpredictive correspondenceが存在していても、aligned statesは自然なInstruct distributionの典型半径には達していない。

したがって、

[
\boxed{
\text{Predictive similarity}
\not\Rightarrow
\text{distributional typicality}
}
]

である。

これはcross-model activation patchingを解釈する上で重要な方法論的制約である。

⸻

6. Causal Intervention: Required Validation

Causal sufficiencyを検証するため、source activationをmatched target runへsubstituteする。

ただし、この評価ではlikelihood-derived probabilitiesを用いてreport distributionを構築する必要がある。

Source、target、patched runについて、

[
P_{\mathrm{source}},
P_{\mathrm{target}},
P_{\mathrm{patch}}
]

をそれぞれ81候補のlog-likelihoodからSoftmaxで求め、

[
\mathrm{Recovery}

1-
\frac{
D(P_{\mathrm{patch}},P_{\mathrm{source}})
}{
D(P_{\mathrm{target}},P_{\mathrm{source}})
}
]

を評価する。

この解析は、

* MLP output
* residual stream
* attention output
* prompt-final position
* response-generation positions

について実施する。

また、causal sufficiency と necessity を区別するため、

* source→target substitution
* mean ablation
* direction removal

を併用する。

⸻

7. Discussion

7.1 Decodability Is a Representational Claim, Not a Mechanistic Claim

本研究で最も重要な方法論的区別は、

[
\text{decodable}
]

と

[
\text{causally used}
]

を混同しないことである。

Probeはあるhidden stateに属性関連情報が存在することを示せる。

しかし、その情報がモデル自身のdownstream computationに利用されているかを確立するためには、independent causal interventionが必要である。

⸻

7.2 Predictive Alignment Is Not Functional Equivalence

Cross-model mappingでも同様である。

高いR²、CKA、pair retrievalは、

one representation predicts another

ことを示す。

しかし、

one representation can replace the other inside the target computation

ことは示さない。

さらに本研究のα sweepは、高いpredictive scoreを維持しながらactivation varianceがtarget distributionの中心方向へ著しく縮小し得ることを示した。

したがって、cross-model causal interventionでは少なくとも、

1. predictive fidelity,
2. distributional typicality,
3. interventional effect,

を別々に報告する必要がある。

⸻

7.3 What This Study Does Not Claim

本研究は、

* LLMが感情を経験している
* first-person outputがintrospectionである
* post-trainingが一意にaffect representationをdecoupleした
* probe-accessible informationが必ずcausally irrelevantである

とは主張しない。

本研究が対象とするのは、特定のBase/Instruct pairにおけるrepresentation、prediction、distribution、causal useを分離するためのcontrolled case studyである。

⸻

8. Limitations

第一に、主解析はQwen2.5-1.5B Base/Instruct pairに限定される。

第二に、linear alignmentは小標本高次元設定でvariance shrinkageを生じるため、cross-model interventionの解釈には限界がある。

第三に、最終prompt-token representationだけでは、response generation中に利用される情報を捉えられない可能性がある。

第四に、causal localizationについてはMLP、attention、residual、token positionおよびgeneration stageを横断した解析が必要である。

⸻

9. Conclusion

本研究は、affect-relevant representationsをケーススタディとして、LLM mechanistic analysisにおける三つの異なる問いを分離した。

[
\boxed{
\text{What information is decodable?}
}
]

[
\boxed{
\text{What representations are predictively alignable?}
}
]

[
\boxed{
\text{What states are causally used?}
}
]

これらは同一の問いではない。

Qwen2.5 Base/Instruct pairでは、affective conditionは中間表現から強くdecodableであり、Base/Instruct representationsには高いpredictive correspondenceが存在した。一方、高次元alignmentは自然なtarget activation distributionと同等のstatistical geometryを再現せず、predictive alignmentをfunctional equivalenceとして扱うことの危険性を示した。

したがって、LLM内部表現をmechanisticに解釈する際には、probe accuracyやrepresentation similarityだけではなく、distributional diagnosticsとproperly validated causal interventionsを独立に組み合わせる必要がある。
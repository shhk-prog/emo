# Post-Training-Associated Changes in the Coupling Between Affective Representations and Self-Reports in LLMs

## 概要 (Abstract)
大規模言語モデル（LLM）は感情認識や情動に配慮した言語タスクを高い精度で実行できる一方で、事後学習（post-training）を経たモデルは自身の感情状態の報告において中立的な回答へ強く収束する傾向を示す。本研究では、この情動的自己報告の中立化が、内部の感情表現の完全な消去（Erasure）、表現の幾何学的変換（Transformation）、出力経路の一様な抑制（Uniform Suppression）、あるいは表現から報告への結合の分散的な変化（Distributed Remapping）のいずれと整合するのかを検証した。語彙交絡を制御した最小対データセット（AIPsy-Affect）と候補列の尤度に基づく評価プロトコルを用い、Qwen2.5-1.5B (Base/Instruct) のペアにおいて対照実験を行った。

検証の結果、Instructモデル内での強制的なActivation Steeringによって制約付き自己報告分布が変化することが確認され、テストした経路が十分に強い介入によって操作可能であることが示された。一方、事前指定したLayer 15 MLPへのAligned Base-to-Instruct patchはBaseに近い自己報告分布を回復せず、探索的なコンポーネントレベルのパッチングでは部分的な回復のみが観測された。テストした条件では、probe-accessible affect-relevant informationの完全消去および最終出力層のみの変化では説明しにくいパターンを観測した。このパターンは、層依存的かつ分散的なrepresentation-to-report couplingの変化と整合する。

---

## 1. 導入 (Introduction)
大規模言語モデル（LLM）は、人間の感情に関わる場面へ急速に導入されている。これに伴いモデルの「共感」や「安全性」が評価されるが、他者の感情状態を推測する能力（認知的共感）と、モデル自身の出力分布がどのように影響を受けるか（機能的情動反応性：affective reactivity）は独立している。

先行研究では、事後学習（SFTやRLHFなど）を経たモデルは「AIであるため感情を持たない」といった定型的な中立値に自己報告が強く収束（greedy collapse）することが指摘されている。本研究の問いは、「LLMが感情を持つか」ではなく、「post-trainingに伴い、感情関連の内部表現と自己報告マッピング（representation-to-report coupling）がどこで・どのように変化するのか」を特定することである。

本研究の最も重要なContributionは、既存研究が「感情関連表現が内部に存在するか」または「内部表現を操作すると行動が変わるか」を主に扱ってきたのに対し、Base/Instruct比較を通じて、affect-relevant representationからconstrained self-reportへの結合そのものを直接検証した点にある。

本稿では以下の3つのリサーチクエスチョン（RQ）に取り組む。
- **RQ1. Representation**: Post-training後もaffect-relevant informationは内部から復元可能か。またBase/Instruct間でそのgeometryはどう異なるか。
- **RQ2. Causal coupling**: 復元可能なaffect-relevant representationはconstrained self-report distributionに因果的影響を持つか。
- **RQ3. Localization**: Base/Instruct間のrepresentation-to-report differenceは、単純なuniform suppression、単一component、またはfinal readout layerで説明できるか。

---

## 2. 課題と競合仮説 (Problem Formulation and Competing Accounts)

事後学習による自己報告の中立化を説明する機序として、本研究では3つの競合仮説（H1〜H3）を設定し、表現の復元性と因果介入に対する反応パターンの違いから識別を試みる。また、これらの結果から導かれる解釈としてDistributed Remapping Account (H4) を検討する。

**表1: 自己報告の中立化に関する競合仮説と識別予測**

| Account | Representation prediction | Report/coupling prediction |
|---|---|---|
| H1: Probe-accessible erasure | Instructでheld-out decodabilityが消失（chance level） | 事後学習後のモデルへの介入効果も消失 |
| H2: Geometry transformation | Direct transferは失敗するが、alignment後にheld-out decodingが回復 | H2単独では指定しない（追加テストが必要） |
| H3: Uniform suppression | Affect-relevant informationは保持 | Coupling gainが広い層・コンポーネントで一様に低下 |
| H4: Distributed coupling change | Affect-relevant informationは保持・変換の可能性あり | 特定・単一のコンポーネントや出力層のみでは説明できない、layer依存の異質な因果効果 |

なお、H2（表現の変換）とH4（結合の分散的変化）は必ずしも排他的ではない。本研究では分析上、表現のGeometry変化（プロービング・クロスモデル予測で評価）と、representation-to-report coupling変化（介入時の自己報告への感度で評価）を概念的に区別する（We conceptually distinguish post-training-associated differences in representation geometry from differences in representation-to-report coupling）。ただし、これらの両項を定量的に一意分解できたと主張するものではない。

---

## 3. 実験設定 (Experimental Setup)

**【定義】** Throughout this paper, "affect-relevant representation" denotes internal information predictive of affective versus matched-neutral context or annotated VA intensity. We do not assume that such representations correspond to subjective affective experience.

### 3.1 Models and datasets

**Model**: Qwen/Qwen2.5-1.5B (Base) and Qwen/Qwen2.5-1.5B-Instruct (Instruct) のペアを用いた。

**1. 外部妥当性確認（EmoBank）**
行動的評価のベースラインとして、人間がValence (不快〜快) と Arousal (沈静〜活性) を付与した大規模コーパス EmoBank を使用した（予備実験用に3,210刺激）。本研究ではEmoBankの **reader perspective** アノテーションを用い、主分析での使用perspectiveを固定する。元の1〜5段階のスコアは線形変換により1〜9のスケールにマッピングして比較を行っている。

**2. 主実験・因果介入（AIPsy-Affect Strict Subset）**
AIPsy-Affect pairsは感情語彙（explicit emotion labels）を意図的に避けて設計されているが、AffectiveとNeutralのナラティブは依然としてイベントの意味論（event semantics）において異なる。したがって、本データセットは明示的な感情語彙による交絡を制御するものであり、全ての語彙的・意味的差異を排除するものではない（controls explicit emotion-label confounds rather than all lexical or semantic differences）。

因果介入実験では、交絡を可能な限り抑えるため、同一の文脈（`pair_id`）に対して `neutral`, `moderate`, `peak` の3段階すべてがアノテーションされているトリプレットのみを抽出した **Strict Matched Subset**（10ペア/30サンプル）を用いた。
- *Peak (Affective)*: "The woman at the front desk asked her to sign for his personal effects... There was no reason to be anywhere in particular. There was no one waiting."
- *Neutral*: "The woman at the front desk asked her to sign for the office supplies... She needed to get back to her desk. There were emails waiting."

**Data Split**: `pair_id` 単位での完全なグループ分割（Strict non-overlapping Train/Dev/Test split）を採用し、同一の最小対が異なるSplitを跨がないように統制した。

### 3.2 Constrained self-report via sequence likelihood
Greedy decodingは同一の最尤JSON列が多くの入力で繰り返し選択される場合、出力分布の段階的変化を隠蔽する。そのため本研究では、単一デコード列ではなく全候補列尤度を測定する **sequence-likelihood-based constrained self-report protocol** を論文全体の共通測定レイヤーとして用いた。プロンプト末尾に続く81通り（$V, A \in \{1..9\}$）のJSON候補の条件付き確率 $P(V, A \mid x)$ を算出する。スコア $s_{v,a}(x)$ は長さ正規化された対数尤度として定義し、温度 $\tau=1.0$ でSoftmax正規化して同時分布と期待値 $E[V \mid x], E[A \mid x]$ を得る。

**Primary Endpoint（一指標）**: **Normalized 2D EMD Recovery**（以下、EMD Recovery）を唯一の主要評価指標とする。これはVAの共同分布全体を捉え、本研究のVA設計と最も一貫する。定義は以下の通りである。

$$\mathrm{Recovery}_{i,\ell} = 1 - \frac{\mathrm{EMD}_{2D}\!\left(P_{\mathrm{patch},i,\ell},\, P_{\mathrm{source},i}\right)}{\mathrm{EMD}_{2D}\!\left(P_{\mathrm{target},i},\, P_{\mathrm{source},i}\right)}$$

ここで $P_{\mathrm{source}}$ はBaseモデルのpeak条件分布、$P_{\mathrm{target}}$ はInstructモデルのneutral条件分布、$P_{\mathrm{patch}}$ はpatch後分布である。**分母が $\epsilon = 0.01$ 未満のpairはprimary analysisから除外**し、除外率を報告する。除外されたpairを含む絶対EMD analysisをsensitivity analysisとして報告する。分布の比較には比率の平均に加えて中央値とpair-level 95% bootstrap CIを併記する。

**Secondary endpoints**: $E[V]$、$E[A]$、$WD_V$（1次元Wasserstein距離）、符号一致率、分布エントロピー。

### 3.3 Probing and alignment
内部表現からの抽出には、プロンプト最終トークン位置（`position = -1`）の残差ストリームを用いた。線形プローブはL2正則化付きのRidge回帰（Valence予測）またはロジスティック回帰（2値分類）を用い、5-fold CVで正則化係数を決定し、独立したTest splitで評価した。Ridge Alignment（H2の検証）では、Train split上のBase/Instruct表現間で線形マッピングを学習し、Alignment-devで調整後、Test splitへ適用した。

### 3.4 Interventions and controls
コンポーネントパッチングでは、対象層のAttentionまたはMLP出力をFull replacement（完全上書き）した。統制パッチとして、Matched-neutral条件の活性化やRandom sourceからのパッチングを用い、アーティファクトを排除した。

### 3.5 Evaluation and statistical analysis
効果の頑健性確認には`pair_id`単位のNon-parametric Bootstrap（1,000回）を用い、95% CIを算出した。層間の多重比較にはFDR補正（Benjamini-Hochberg法）を適用した。H3の検証用混合効果モデルでは、層と刺激強度を固定効果、`pair_id`をランダム効果としてモデリングした。

---

## 4. 結果 (Results)

### 4.1 Behavioral neutralization
**【目的】** アライメントを受けたLLMにおいて、テキストの感情的性質の「認識」と「制約付き自己報告」が乖離している現象を確認する。
**【結果】** 感情認識タスクでは、モデル出力と人間アノテーションとの間に強い正の相関（Valence $r=0.921$）が確認された。一方、greedy decodingによる自己報告タスクでは、3,210回の試行の約98.6%が `{"valence": 5, "arousal": 5}` に収束した。Sequence-likelihood protocolによるQwen2.5-1.5B (Instruct) の制約付き自己報告分布の期待値も中立（Valence=5.0付近）へ強く偏り、この中立化はTemperature Scaling（$\tau=0.2 \dots 5.0$）に対してロバストであった。

### 4.2 Retained probe-accessible information
**【目的】** 自己報告の中立化が内部空間からの感情情報の完全消失（H1）によるものかを検証する。
**【結果】** Instructモデルの中間層（Layer 14–25）残差表現から、感情刺激 vs 中立対照の二値分類においてROC-AUC > 97.5%、Valenceの連続値予測においてheld-out $R^2=0.318$ という高い性能が得られた。
**【示唆】** テストされた隠れ状態において、Affective/Neutral条件を線形に区別する情報が完全に消失しているという説明（H1）は支持されなかった。

### 4.3 Geometry transformation
**【目的】** 情報は消去されていないが、表現空間の幾何学的変換（H2）が生じているかを検証する。
**【結果】** BaseモデルのプローブをInstructモデルの表現に直接適用（Direct Transfer）した場合は予測性能が著しく低下（$R^2 \approx -100$）した。しかし、Ridge Alignmentを適用すると、コントロール（Token count等 $R^2 \approx 0.05\text{–}0.12$）と比較してValence（$R^2 \approx 0.58$）の予測性能が顕著に回復した（表2）。

**表2: 厳密分割での交差デコーディング ($R^2$) — Valence**
| Layer | Direct Transfer | Ortho. Procrustes | Ridge Alignment |
|-------|-----------------|-------------------|-----------------|
| 12    | -132.80         | 0.03              | **0.51**        |
| 16    | -93.81          | -0.11             | **0.55**        |
| 20    | -97.35          | -0.01             | **0.59**        |

**【示唆】** 情動情報は保持されているが、事後学習により非直交的かつ部分的に線形整列可能な空間の歪み（Representation Transformation, H2）を受けている証拠を得た。ただし、Ridge alignmentでデコーディングが回復することはH2（geometryの変換）の証拠であり、aligned representationが因果的に代替可能か（causal substitutability）はH2単独では指定されない追加テストである。

### 4.4 Dissociation Between Internal Decodability and Likelihood-Based Self-Report Sensitivity
**【目的】** 内部表現の刺激感度（probe decodability）と、likelihood-basedな制約付き自己報告の刺激感度が、均一な関係にあるかを記述的に比較する。

**【記号の定義】** 本節では以下の記号を用いる。
- $r^{\mathrm{probe}}_{\ell,m}$ = $\mathrm{corr}\!\left(\hat{V}^{\mathrm{probe}}_{\ell,m}(x),\, V^{\mathrm{human}}(x)\right)$（層 $\ell$、モデル $m$ の内部プローブ感度）
- $r^{\mathrm{report}}_{m}$ = $\mathrm{corr}\!\left(E_m[V \mid x],\, V^{\mathrm{human}}(x)\right)$（モデル $m$ のlikelihood-based自己報告感度）

**【注意】** $r^{\mathrm{report}}_{m}$ はsequence-likelihood $E[V]$（自己報告出力）と人間VADラベルの相関であり、自己報告出力以外の独立した下流行動指標ではない。したがって本節は「report–behavior dissociation」を示す分析ではなく、内部decodabilityとlikelihood-based自己報告感度の**記述的比較**である。

**【結果】** 表3に示すように、テストしたBase/Instruct条件でLayer 16における$r^{\mathrm{probe}}$と$r^{\mathrm{report}}$の差の符号が逆転した。Baseでは$r^{\mathrm{probe}}_{16} = 0.479 > r^{\mathrm{report}} = 0.365$（差 = +0.114）であったのに対し、Instructでは$r^{\mathrm{probe}}_{16} = 0.464 < r^{\mathrm{report}} = 0.629$（差 = −0.165）であった。なお、InstructでHBC_V（=0.629）がBaseより高い値を示すことは、greedy collapseが生じていても尤度空間では人間VAとの相関が保持されていることを示す。これは追加の考察を要する発見である。

**表3: 内部プローブ感度 $r^{\mathrm{probe}}$ とlikelihood-based自己報告感度 $r^{\mathrm{report}}$ の比較 — Layer 16, Valence**
| モデル | $r^{\mathrm{probe}}_{16}$ | $r^{\mathrm{report}}$ | 差 ($r^{\mathrm{probe}} - r^{\mathrm{report}}$) |
|--------|---------------------------|----------------------|-------------------------------------------------|
| Base     | 0.479                     | 0.365                | **+0.114**                                      |
| Instruct | 0.464                     | 0.629                | **−0.165**                                      |

また、終盤層（Layer 27）では $r^{\mathrm{probe}}_{27,\mathrm{Base}} = 0.259$ に対し $r^{\mathrm{probe}}_{27,\mathrm{Instruct}} = 0.131$（差 = +0.128）であり、Baseモデルが終盤層においてより高い内部刺激感度を持つことが確認された。

**【結論】** Across layers and model variants, internal representational sensitivity and likelihood-based constrained self-report sensitivity did not exhibit a uniform relationship. In particular, the sign of $r^{\mathrm{probe}} - r^{\mathrm{report}}$ differed between the tested Base and Instruct conditions at Layer 16. This descriptive result motivates the later causal intervention analyses but does not constitute an independent behavioral validation.

### 4.5 Decodability does not guarantee causal substitutability
**【目的】** 表現のdecodability回復がcausal substitutabilityを保証するか（Aligned cross-model patching）、また強制介入によって経路自体が保持されているかを確認する（Activation Steering）。

**Pre-specified aligned patch**
Train/Dev/Testの厳密な3分割上でRidge Alignmentを学習し、Baseモデルのpeak時の内部状態をInstruct空間へ写像した後、InstructモデルのLayer 15 MLPへ全量上書きした（本実験では事前に公開したconfigファイルとcommit hashによりLayer 15 MLPをprimary patchターゲットとして特定；OSFによる時間記録済み事前登録は行っていないため、以後「pre-specified」と表記する）。表現レベルでは交差デコーディング性能（$R^2 \approx 0.58$）が回復していたにもかかわらず、最終的な自己報告出力（$E[V]$）の分布シフトは実質的に0であった（表4）。

**表4: Aligned Cross-Model Patching (Pre-specified: Layer 15 MLP, Strict Test Set)**
| Quantity | Base source (Peak) | Instruct target (Neutral) | Aligned-patched |
|---|---:|---:|---:|
| Expected Valence ($E[V]$) | 6.65 | 5.37 | 5.38 |
| $WD_V$ to Base source | 0.00 | 0.82 | 0.81 |
| 2D EMD Recovery (primary) | — | 0.0% | ~0.0% |

**Exploratory component sweep（事後探索）**
事前指定のLayer 15 MLPが0効果であったため、事後的（post-hoc）に全層のAttentionとMLPを対象としたスイープを実施した。Layer 10 MLP や Layer 14 AttentionへのAligned cross-model patchingが部分的な回復（$\Delta WD_V \approx -0.18$）を示したが、これらはpost-hoc selectionであるため多重比較の問題をはらんでいる。したがって、独立したテストセットでの検証を要する仮説生成的な結果として位置づける。

The pre-specified Layer-15 aligned patch produced essentially no recovery. The exploratory layer/component sweep subsequently identified partial effects at Layer-10 MLP and Layer-14 attention, but these were selected post hoc and are therefore treated as hypothesis-generating.

**Activation Steering**
極端な介入強度（Forced activation regime）でのActivation Steeringにおいては中立値からの脱却が観測されたため、tested directionに沿った制約付き自己報告分布の操作が可能であることが示された。ただし、これは強制介入の文脈での操作可能性を示すにすぎず、自然な感情文脈の情報が通常の計算経路を経て自己報告へ因果寄与することを確立するものではない（Forced activation steering demonstrates manipulability; it does not establish that naturally occurring affective variation uses the same direction or pathway）。

**【示唆】** decodability restored ≠ causal substitutability restored。また、探索スイープ結果は aligned representation alone $\not\Rightarrow$ uniform causal substitutability ながら、$\exists l, c$: partial causal effect であり、coupling がlayer-dependentであることを示唆する（H4への合流的証拠）。

### 4.6 No clear evidence for uniform suppression
**【目的】** 事後学習による中立化が、全層・全コンポーネントで一様なcoupling低下（H3: Uniform Suppression）で説明できるかを検証する。

**【混合効果モデル】** H3の検証として、各層 $\ell$ ごとに独立したモデルをfitした。刺激 $i$（Base/Instruct両条件分を積む）、モデル種別 $m \in \{\mathrm{Base}, \mathrm{Instruct}\}$、pair_idをランダム効果として以下のモデルを層ごとに推定した。

$$E[V]_{i,m,\ell} = \beta_{0,\ell} + \beta_{1,\ell}\, z_{i,m,\ell} + \beta_{2,\ell}\, \mathbb{1}[m=\mathrm{Instruct}] + \beta_{3,\ell}\, z_{i,m,\ell}\,\mathbb{1}[m=\mathrm{Instruct}] + u_{\mathrm{pair}(i)} + \epsilon_{i,m,\ell}$$

ここで $z_{i,m,\ell}$ は層 $\ell$ のプローブ予測値（$m$ で条件付き）、$u_{\mathrm{pair}(i)}$ はpair_idのランダム効果である。なお、Base/Instructは各1チェックポイントの比較であり、モデル種別は固定効果として扱う。実際のpost-training手続きの完全な内容は外部研究者には観測不能なため、$\mathbb{1}[m=\mathrm{Instruct}]$ はpost-training手続き全体の因果効果変数としてではなく、公開済みチェックポイント間の差異指標として解釈すること。

**表5: 混合効果モデル（層別fit） — $\beta_{3,\ell}$（交互作用項; coupling gain変化）**
| Layer | Valence $\beta_{3,\ell}$ | $p$-value | 95% CI | Arousal $\beta_{3,\ell}$ |
|-------|--------------------------|-----------|--------|---------------------------|
| 12    | +0.752                   | 0.027*    | [+0.085, +1.419] | −0.832 |
| 16    | +0.621                   | 0.525     | [−1.293, +2.536] | −0.567 |
| 20    | +0.443                   | 0.541     | [−0.977, +1.862] | −0.229 |
| 24    | −0.028                   | 0.940     | [−0.762, +0.705] | −0.096 |
| 27    | +0.223                   | 0.383     | [−0.279, +0.726] | −0.016 |

*\* FDR補正前。FDR補正後（BH法、5層×2次元）有意なし。*

**【示唆】** $\beta_{3,\ell}$の符号は層によって正負が混在し（Layer 12でValenceは正、ArousalはFDR後非有意）、FDR補正後に有意な一様な負の変化は確認されなかった。The data do not provide clear evidence for a uniform negative coupling shift（H3を反証したのではなく、H3の明確な証拠が得られなかった）。

### 4.7 Localizing coupling differences
事後学習による中立化が、特定の単一コンポーネントや最終出力層のみで説明できるかを検証する。

**No single-component rescue**: Section 4.5 の探索スイープで最大効果を示したLayer 10 MLP / Layer 14 Attentionへの介入でも、Base型の分布を完全には回復できなかった。

**No simple late-residual shortcut**: 特定コンポーネント（attn_14等）の寄与を最終Transformerブロック直前の入力（$h_{26}$）に局所置換（Late-Residual Substitution）したところ、Base型への回復を全く示さなかった。

**Final output layers do not explain the difference**: 最終層のResidual表現、RMSNorm、Unembedding (lm_head) の重みをBase/Instruct間で交差させた8条件スワップ解析を実施した。分布のシフトはlm_headのソース（$\Delta WD_V \approx 0.01$）よりもResidual表現のソース（$\Delta WD_V \approx 0.20$）に強く依存した。

---

## 5. 考察 (Discussion)

本研究は、自己報告と内部表現の解離現象について以下の3点の限定的証拠（controlled mechanistic case study）を提供する。

1. **測定の分離**: 制約付き数値自己報告を離散出力ではなく候補列尤度分布として測定することで、greedy出力の完全収束と確率分布上の因果的反応性を分離した。
2. **情報保持と経路の部分的保持**: post-training後もprobe-accessibleなaffect-relevant informationは残存し、強制介入によってself-report distributionを変化させることが可能であった。
3. **結合マッピングの変化機序**: 自己報告の中立化現象は、情報の消去や出力直前のreadoutのみの更新とは整合せず、層・コンポーネント依存の representation-to-report coupling の分散的変化（distributed remapping）と最も整合する。

**Evidence Consistent with a Distributed Coupling-Change Account**

単一の実験ではなく、以下の複数の結果のパターンによって、分散的結合変化accountと整合する証拠が提示される。**なお、項目3、6、7、8は「単一箇所説」への**negative result**であり、Distributed Remappingの**積極的証拠**ではなく、単純代替説の制約として統一的に解釈すること。**

1. 内部表現のaffect-relevant informationは残存する（Section 4.2）
2. 表現のgeometryは変化している（Section 4.3）
3. 「aligned representationが数値的に回復してもself-reportは戻らない」はnegative result—decodability ≠ causal substitutability（Section 4.5）
4. 一部のcomponentにはpartial causal effectが存在する（Section 4.5 exploratory）
5. componentごとに効果量が異なる（Section 4.5, 4.7）
6. single componentではfull rescueしない (negative result)（Section 4.7）
7. late shortcutも失敗する (negative result)（Section 4.7）
8. lm_head/RMSNorm swapも説明不足 (negative result)（Section 4.7）

No single experiment uniquely identifies a distributed remapping mechanism. Rather, this account is motivated by the joint pattern across representation alignment, causal substitution, component-wise interventions, and output-layer swaps.

**表6: Evidence Map — 段階別の観測結果と結論**
| 段階 | 観測 | 言えること |
|---|---|---|
| Behavioral | Instruct greedy self-reportがneutral化 | greedy履歴でのreport sensitivityが低い |
| Representation | Within-model probe成功（AUC>0.975） | complete probe-accessible erasureではない |
| Geometry | Ridge alignmentでheld-out decodingが回復 | non-orthogonal linear transformationと整合 |
| Internal vs Report sensitivity | $r^{\mathrm{probe}}$と$r^{\mathrm{report}}$の差の符号がBase/Instructで逆転（Layer 16） | 均一な関係ではない（記述的） |
| Aligned patch (pre-specified L15) | decodability回復でもreportは回復しない | decodability ≠ causal substitutability (negative) |
| Aligned patch (exploratory L10/L14) | 部分的回復（$\Delta WD_V \approx -0.18$） | 一部couplingは保持；post-hoc選択 |
| Steering | 強制介入でreportが変化 | tested direction内での操作可能性を示す |
| Mixed model | $\beta_{3,\ell}$の正負が混在、FDR後有意なし | uniform suppressionの明確な証拠なし |
| Component patch | single rescueなし (negative) | single bottleneckでは説明不足 |
| Late residual | rescueなし (negative) | simple shortcutでは説明不足 |
| Output swap | residual sourceが支配的 (negative) | lm_head/RMSNormのみでは説明不足 |

**中心結論**: The evidence constrains simple alternatives: it is inconsistent with complete probe-accessible erasure and does not support an explanation confined to the final output layers. The remaining pattern is compatible with distributed, layer-dependent coupling changes, but does not uniquely identify their mechanism.

---

## 6. 制限 (Limitations)
- **データとモデルの限界**: 主分析は10対のAIPsy-Affect厳密トリプレットと、単一モデルファミリー（Qwen2.5-1.5B）の1ペアの比較に依存している。N=10は厳密に管理された条件定め実験として有効であるが、結果はこのスコープ内のcontrolled case studyとして解釈すること。一般化にはより大規模なデータペアおよび複数アーキテクチャによる検証が必要である。pairごとの結果全体をdot plotで可視化し、平均値と中央値の両方を報告する。
- **Section 4.4の記述的制限**: Section 4.4で用いた$r^{\mathrm{report}}$（sequence-likelihood $E[V]$と人間VADの相関）は自己報告出力以外の独立した行動指標ではない。したがって、本分析は内部decodabilityとlikelihood-based自己報告感度の記述的比較にとどまり、introspective faithfulnessや主観的状態への同定を直接に確立するものではない。
- **Aligned patchingに伴う分布外活性**: 表現変換後も完全に分布外（OOD）の活性が排除できたかは保証されない。decodabilityの回復（$R^2$回復）が因果的代替可能性を保証するわけではない。

---

## 7. 結論 (Conclusion)
本研究は、LLMにおける事後学習が自己報告の中立化をもたらす機序を調査した。テストした条件では、probe-accessible affect-relevant informationの完全消去および最終出力層のみの変化では説明しにくいパターンを観測した。このパターンは、層依存的かつ分散的なrepresentation-to-report couplingの変化と整合する。この結果は、LLMの情動的反応性とその評価において、単純な出力の制約にとどまらない内部マッピングの変化を考慮する必要性を強調するものである。

---

## Appendix (補遺)

### Appendix A: Sequence likelihood implementationおよび指標定義
81通り（$V, A \in \{1..9\}$）のJSON文字列の条件付き確率 $P(V, A \mid x)$ を算出し、スコアは長さ正規化された対数尤度として計算した上で、温度 $\tau=1.0$ でSoftmax正規化して同時分布を導出した。内部プローブ感度とlikelihood-based自己報告感度の定義は次の通りである。
- $r^{\mathrm{probe}}_{\ell,m} = \mathrm{corr}\!\left(\hat{V}^{\mathrm{probe}}_{\ell,m}(x),\, V^{\mathrm{human}}(x)\right)$：層 $\ell$、モデル $m$ の内部プローブ予測値と人間VADラベルのPearson相関。
- $r^{\mathrm{report}}_{m} = \mathrm{corr}\!\left(E_m[V \mid x],\, V^{\mathrm{human}}(x)\right)$：モデル $m$ のsequence-likelihood $E[V]$と人間VADラベルのPearson相関。これは自己報告出力から作った指標であり、自己報告以外の独立した行動指標ではない。

### Appendix B: Probe and alignment details
Train（プローブ学習）、Alignment-dev（写像の適合）、Held-out test（評価）の厳密な3データ分割を使用し、情報漏洩を防いだ。プローブはL2正則化（Ridge）を用い、5-fold CVで正則化係数を最適化した。

### Appendix C: Detailed Experiment Specifications
- **Model**: Qwen/Qwen2.5-1.5B (Base) and Qwen/Qwen2.5-1.5B-Instruct (Instruct).
- **Hardware/Framework**: NVIDIA H200 (1 GPU), FP16/bfloat16. `torch>=2.4`, `transformers>=4.45`.
- **Intervention Position**: Activation patching and steering were applied to `position = -1` (final prompt token).
- **Batching**: Left Padding, batch size 512.
- **Strict subset**: 10 pairs (30 samples) with all three intensity levels (neutral/moderate/peak) per pair_id.

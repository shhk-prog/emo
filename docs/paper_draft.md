# 大規模言語モデルの情動反応性評価  
## Russellの感情円環に基づく情動軌道と機能的内部表現の段階的検証

## Abstract

Large language models (LLMs) can generate language that appears emotionally appropriate or empathetic. However, it remains unresolved whether these outputs arise solely from surface-level linguistic regularities or whether they are associated with input-sensitive internal representations that functionally influence subsequent behavior. This study proposes a reproducible framework for evaluating LLM affective reactivity using Russell’s valence–arousal (VA) space.

The framework distinguishes four outputs obtained in independent sessions: a baseline model-reported affective state, recognition of the affective position of a stimulus, a model-reported affective response elicited after processing the stimulus, and an empathic free-text response. LLM-reported VA values are elicited on a 1–9 scale and normalized to \([-1,1]\) for geometric analysis. The primary behavioral outcome is the displacement from baseline to post-stimulus VA, represented as a two-dimensional reaction vector.

The preliminary experiment evaluates whether these reported reaction vectors covary systematically with human-annotated affective properties of stimuli in EmoBank. It measures recognition accuracy, valence and arousal reactivity, response magnitude, stimulus-direction alignment, post-state distance to the stimulus anchor, and stability across repeated generations. Importantly, EmoBank is used as an affective anchor for stimuli, not as direct ground truth for a human observer’s pre-to-post affective trajectory.

The planned main experiment, to be conducted only for open-weight models, will test whether behavioral reactivity is accompanied by functionally relevant internal representations. It will combine hidden-state probing, representational similarity analysis, activation patching, ablation, and steering. Lexical-emotion, contextual-emotion, neutral, label-shuffled, and paraphrase controls will be used to distinguish contextual affect processing from superficial dependence on emotion words.

The proposed framework does not test whether LLMs have subjective feelings, consciousness, or human-like emotional experience. It instead evaluates **functional affective reactivity**: a model-reported behavioral response that is stimulus-sensitive and, where supported by internal intervention experiments, associated with causally behavior-relevant internal representations.

## Keywords

Large language models; affective reactivity; affective empathy; Russell circumplex model; valence; arousal; EmoBank; emotion recognition; mechanistic interpretability; activation patching.

***

# 1. Introduction

大規模言語モデル（LLM）は、悲しみ、不安、怒り、喜びを含む文脈に対し、慰め、承認、励まし、支援的な言語を生成できる。このため、LLMはしばしば「共感的」と表現される。しかし、共感的に見える文を生成できることと、他者の感情的入力に応じてモデル内の処理状態が変化していることは、理論的にも実証的にも別の主張である。

人間研究では、他者の感情を推論する**認知的共感**と、他者の感情に接して自身の情動状態が変化する**情動的共感**または情動反応性が区別される。本研究は、このうち後者に対応する現象をLLMで操作的に検討する。ただし、本研究はLLMの主観的経験を直接測定できない。そのため、測定対象を「標準化されたプロンプト条件において、感情的刺激の前後で生じるモデル報告VA値の変位」と明確に限定する。

既存研究の多くは、感情ラベルの認識、感情分類、共感的対話生成、あるいは人間による共感応答品質の評価に焦点を当ててきた。これらは重要だが、LLMが他者の感情的経験を処理した後に、どのような反応的変位を示すかを直接には測らない。

EmotionBenchは、appraisal theoryに基づく情動喚起状況を提示し、LLMの感情スコアの前後変化を人間結果と比較した代表的研究である。400以上の状況と1,200人超の人間評価を用い、LLMが一部の状況に適切に応答する一方、人間の情動行動との整合性には限界があることを示した。 本研究は、刺激提示前後の差分を測定するという基本的発想を継承する。 [arxiv](https://arxiv.org/abs/2308.03656)

一方、本研究はRussellの感情円環に基づく連続的なValence–Arousal（VA）空間を用い、LLMの反応を単一スコアの変化ではなく、二次元空間における**情動反応ベクトル**として分析する。さらに、刺激の感情を推定するRecognitionと、刺激処理後にモデルが報告するAffective Receptionを独立セッションとして実行し、感情認識と反応性を混同しない。

本研究は二段階で構成される。予備実験では、EmoBankの人間注釈済みVA値を刺激側の情動アンカーとして用い、LLMのモデル報告情動反応が刺激特性に応じて系統的に変化するかを行動的に評価する。計画中の本実験では公開重みモデルに限定し、隠れ状態のプロービングと因果的内部介入を通じて、行動的な反応が機能的内部表現に媒介されるかを検証する。

***

# 2. Research Questions

**RQ1: 感情認識**  
LLMは、EmoBankにおいて人間が注釈した刺激のValenceおよびArousalをどの程度推定できるか。

**RQ2: 情動反応性**  
LLMは、他者の感情的経験を表す刺激の処理後に、モデル報告VAをBaselineから系統的に変位させるか。

**RQ3: 刺激情動との対応**  
LLMの反応ベクトルは、刺激側の人間VAアンカーと、Valence・Arousal・方向・反応強度の観点でどの程度対応するか。

**RQ4: 認識と反応の分離**  
刺激感情を正確に認識するモデルは、より大きく、あるいはより整合的な情動反応を示すのか。それとも、RecognitionとAffective Receptionは独立した能力か。

**RQ5: 語彙依存性**  
モデルの反応は明示的感情語の存在に依存するか。それとも、感情語を含まない状況的・文脈的手がかりに対しても維持されるか。

**RQ6: 機能的内部表現**  
公開重みモデルにおいて、刺激VAおよびモデル報告VA変位を隠れ状態から復元できるか。また、その表現への因果的介入は後続のモデル報告VAおよび応答生成を変化させるか。

***

# 3. Contributions

本研究の貢献は以下の通りである。

1. LLMの情動的反応性を、RussellのVA空間におけるBaselineからPostへの二次元変位ベクトルとして定量化する。  
2. 感情認識、刺激誘発性のモデル報告情動反応、共感的自由応答を独立した条件として収集し、測定上分離する。  
3. EmoBank等の既存の人間アノテーション済みコーパスを、刺激側の情動アンカーとして再利用する再現可能な評価手順を示す。  
4. 感情語あり・感情語なし・中性・ラベルシャッフル・パラフレーズを用い、語彙的追随と状況意味に基づく反応を分けて評価する。  
5. 行動的な出力評価と、公開重みモデルにおけるプロービング・RSA・patching・ablation・steeringを段階的に接続し、機能的内部表現に関する検証可能な仮説を提示する。  

***

# 4. Method

## 4.1 Operational Definitions

本研究では、「LLMの感情」という表現を避け、以下を区別する。

| 用語 | 操作的定義 |
|---|---|
| Stimulus affect | EmoBankの人間注釈に基づく、刺激文のVA位置 |
| Recognized affect | LLMが平均的読者または刺激中の人物について推定するVA |
| Model-reported affective state | 定型プロンプトに対してLLMが数値JSONとして報告するVA |
| Affective reactivity | BaselineとAffective Receptionの間のモデル報告VA変位 |
| Empathic response | 刺激に対して生成された自由テキスト応答 |
| Functional internal affective representation | 内部プローブで復元可能であり、介入により後続出力へ因果的影響を与える感情関連表現 |

従って、本研究の行動的測定は、主観的感情経験ではなく、`self-reported affective state`、`elicited affective response`、および`behavioral proxy for affective empathy`を対象とする。 

## 4.2 Experimental Conditions

各刺激 \(x_i\) に対して、以下を**独立したAPIセッションまたは独立した会話コンテキスト**で実行する。

| 条件 | 出力 | 用途 |
|---|---|---|
| empty_baseline | 刺激なしのモデル報告VA | モデル・反復ごとの基準状態 |
| recognition_va | 刺激の推定VA | 感情認識能力 |
| post_reported_va | 刺激処理後のモデル報告VA | 刺激誘発性反応 |
| free_response | 自由文応答 | 対人的共感表現の補助評価 |

recognition_vaの出力をpost_reported_vaの会話履歴へ渡さない。この分離により、モデルが自ら出力した認識値に後続回答をアンカリングすることを避ける。 
## 4.3 Scales and VA Representation

LLMにはValenceおよびArousalを1–9尺度で出力させる。解析では、モデル報告値 \(x^{(9)}\) を以下の変換で \([-1,1]\) に正規化する。

\[
x_{\mathrm{norm}}=\frac{x^{(9)}-5}{4}
\]

刺激 \(i\)、モデル \(m\)、条件 \(c\) のVA状態を、

\[
\mathbf{e}_{m,i}^{c}
=
(v_{m,i}^{c},a_{m,i}^{c})
\]

と定義する。

Baselineからpost_reported_vaへの変位ベクトルは、

\[
\Delta\mathbf{e}_{m,i}
=
\mathbf{e}_{m,i}^{\mathrm{post}}
-
\mathbf{e}_{m,i}^{\mathrm{baseline}}
\]

である。各成分は、

\[
\Delta v_{m,i}
=
v_{m,i}^{\mathrm{post}}-v_{m,i}^{\mathrm{baseline}}
\]

\[
\Delta a_{m,i}
=
a_{m,i}^{\mathrm{post}}-a_{m,i}^{\mathrm{baseline}}
\]

として計算する。反応量は、

\[
R_{m,i}
=
\sqrt{(\Delta v_{m,i})^2+(\Delta a_{m,i})^2}
\]

とする。

### 反応量の離散性

1–9の整数尺度では、正規化後の最小の非ゼロ変化は片軸で0.25である。したがって、\(R<0.10\)を「実質的無反応」と解釈することは尺度分解能と整合しない。

主解析では、以下を報告する。

- \(R=0\) の完全無反応率  
- \(R>0\) の反応率  
- \(R\) の中央値、平均、四分位範囲  
- 感度分析として \(R<0.125\) および \(R<0.25\) を閾値とした結果

## 4.4 Human Affective Anchor

EmoBankの人間注釈値を、刺激 \(x_i\) の情動アンカー

\[
\mathbf{h}_{i}
=
(v_i^H,a_i^H)
\]

として使用する。EmoBankが1–5尺度である場合、以下の変換を使用する。

\[
v_i^H=\frac{V_i-3}{2},
\qquad
a_i^H=\frac{A_i-3}{2}
\]

しかし、EmoBankにおいてアノテーション視点（読者視点か筆者視点か）が異なる場合があるため、分析時はメタデータに基づき `reader-perspective` 列を明示的に指定して変換を行う。

また、\(\mathbf{h}_{i}\) は刺激の人物感情または読者が知覚する刺激の情動的位置であり、刺激を読んだ人間の事前・事後の情動変位の正解値ではない。したがって、本研究はEmoBankのみから「LLMの情動軌道が人間の情動軌道と一致する」とは主張しない。

## 4.5 Behavioral Evaluation Metrics

### Recognition Accuracy

Recognition条件のモデル推定値と刺激アンカーの整合性を、Valence・Arousal別に評価する。

- Spearmanの順位相関
- Pearson相関
- MAE
- RMSE
- Concordance correlation coefficient
- 象限一致率

### Reactivity–Stimulus Association

post_reported_vaにおいて、\(\Delta v\)と刺激側Valence \(v_i^H\)、\(\Delta a\)と刺激側Arousal \(a_i^H\) の関係を推定する。

\[
\Delta y_{m,i}
=
\beta_0
+
\beta_1 y_i^H
+
\beta_2 \mathrm{Model}_m
+
\beta_3 y_i^H \times \mathrm{Model}_m
+
u_i
+
\epsilon_{m,i}
\]

ここで \(y\) はValenceまたはArousal、\(u_i\) は刺激に関するランダム切片である。主たる推論は、刺激VAに対する反応傾き \(\beta_1\) およびモデル間の傾き差 \(\beta_3\) である。

### Anchor Direction Alignment

刺激アンカーの原点からの方向と、LLM反応ベクトルの方向の角度的一致を補助指標として算出する。

\[
\mathrm{ADA}_{m,i}
=
\frac{
\Delta\mathbf{e}_{m,i}\cdot\mathbf{h}_{i}
}{
\left\|\Delta\mathbf{e}_{m,i}\right\|_2
\left\|\mathbf{h}_{i}\right\|_2
}
\]

ADAは「刺激の情動位置と同じ方角へLLMが反応する傾向」を表す。これは人間の情動軌道との一致度ではないため、**empathic alignment**の主要指標ではなく、`stimulus-direction alignment`として報告する。

### Post-state Distance to Anchor

post_reported_va後のモデル報告VAと刺激アンカーの距離を補助的に算出する。

\[
D_{m,i}
=
\left\|
\mathbf{e}_{m,i}^{\mathrm{post}}
-
\mathbf{h}_{i}
\right\|_2
\]

この値は、モデルが刺激側の情動位置にどの程度近い位置を報告するか、すなわち情動的同調・模倣の程度を示すものである。適切な共感応答の正解距離ではない。

### Stimulus-to-response Gain

\[
G_{m,i}
=
\frac{R_{m,i}}{\left\|\mathbf{h}_i\right\|_2+\epsilon}
\]

\(G_{m,i}\) は刺激情動の中心からの偏位量に対して、LLMの反応量がどの程度大きいかを示す。人間の反応量で規格化していないため、`Empathic Gain`ではなく、**stimulus-to-response gain**と表記する。

### Structural Analysis

刺激空間とモデル反応空間の構造的一致を、以下で評価する。

- 刺激間距離行列と反応ベクトル間距離行列のRSA
- Procrustes analysis
- CCA
- 象限分類のmacro-F1
- Valence・Arousal別の効果量と信頼区間

## 4.6 Control Conditions

| 条件 | 定義 | 検証対象 |
|---|---|---|
| Original | EmoBankの原文 | 主解析 |
| Lexical-emotion | 明示的感情語を含む刺激 | 感情語への追随 |
| Contextual-emotion | 明示的感情語を含まない刺激 | 文脈意味に基づく反応 |
| Neutral | VA原点近傍の刺激 | 非情動的変動 |
| Label-shuffled | 分析時に刺激–VA対応を無作為化 | 偶然相関の統制 |
| Paraphrase | 意味保存パラフレーズ | 表層形式・プロンプト頑健性 |

Label-shuffledは新たなモデル入力条件ではない。人間注釈と刺激の結び付きを分析段階で破壊する置換検定のための統制である。

## 4.7 Prompting and Output Validation

主解析では、固定system prompt、固定ユーザープロンプト、temperature = 0.0、top-p = 1.0、固定最大生成長を用いる。各プロンプトはハッシュ化し、`run_id`、`prompt_id`、`system_prompt_hash`、`prompt_hash`、デコーディング設定、応答本文、パース状態とともに保存する。 

出力の取得においては、ValenceとArousalの1–9整数尺度を要求し、スキーマレベルで整数以外の入力を弾く。無効JSON、範囲外値、空応答、拒否、API失敗は削除せず、`parse_status`および`failure_reason`とともに保存する。Valid JSON Rate、Scale-valid Rate、条件別の拒否率・失敗率も報告する。

## 4.8 Reproducibility

現行パイプラインは、`run_id`、prompt hash、Git commit、設定ファイルのスナップショット、実行時刻、応答本文、パース結果をJSONLとして保存する再現性設計を備える。 本論文では、以下を各実験結果と共に公開・保存する。

- データセット名、配布版、ライセンス、ファイルハッシュ
- 刺激抽出規則、層別抽出の乱数シード、除外規則
- モデル提供者、モデルID、モデルrevisionまたはsnapshot
- system prompt・user prompt・ハッシュ
- temperature、top-p、max tokens、seed、retry規則
- 実験コードのGit commit、依存関係lock file、実行環境
- 生のJSONL応答、派生データ、解析スクリプト、図表生成スクリプト

ただし、本実験の内部表現抽出・因果介入モジュールは現在計画・開発段階である。予備実験向けのAPIクライアント・検証ロジックは実装済みである。
***

# 5. Experiments

## 5.1 Preliminary Experiment: Behavioral Affective Reactivity

### Objective

予備実験の目的は、LLMが人間注釈済みの刺激情動に対して、empty_baselineからpost_reported_vaへ系統的なモデル報告VA変位を示すかを検証することである。

この実験では、モデルが主観的感情を持つこと、または人間と同一の内部状態変化を持つことを主張しない。示せるのは、固定された刺激・プロンプト・生成設定のもとで、モデル報告VAが刺激特性に依存して変化するという行動的事実である。

### Models

- クローズドモデル：APIを通じて利用可能なモデル  
- 公開重みモデル：Llama、Qwen、Gemma等のInstructモデル  
- 追加比較：対応するbaseモデル  

クローズドモデルについては行動的結果のみを報告する。隠れ状態を取得できないクローズドモデルに関して、内部状態の存在や因果的機能について結論しない。

### Sampling

EmoBank刺激をVA平面の格子に層別化し、各セルから同数または最大数を無作為抽出する。現行実装は3×3のVA格子とセルごとの抽出数を設定できる。論文では、セル境界、各セルの候補数、採択数、除外数、抽出seedを明記する。
### Procedure

各モデル・刺激・反復について、以下を別セッションで実施する。

1. empty_baseline VA
2. recognition_va
3. post_reported_va
4. free_response

主解析はtemperature = 0.0で実施する。頑健性分析ではtemperature = 0.3および0.7を追加し、各刺激を複数回反復する。temperature = 0でもAPI側の非決定性があり得るため、温度0での反復結果も収集する。

### Statistical Analysis

ValenceとArousalを独立に扱い、以下を報告する。

- Recognitionと刺激アンカーの相関・誤差
- \(\Delta V\)、\(\Delta A\)、\(R\)の分布
- 刺激Valenceと\(\Delta V\)、刺激Arousalと\(\Delta A\)の混合効果モデル
- ADA、Post-state Distance、Stimulus-to-response Gain
- 感情語あり・なし条件における効果量差
- 中性刺激でのベースライン変動
- シャッフル統制に対する置換検定
- モデル・温度・プロンプト言い換え間の再現性

Valence-dominant性は、ValenceとArousalの効果量の差をブートストラップ信頼区間で評価する。Valence相関だけが高くても、Arousal側の効果と統計的に比較しない限り、Valence-dominantとは結論しない。

## 5.2 Main Experiment: Functional Internal Representation

### Objective

本実験では、予備実験で観測された行動的反応性が、公開重みLLMの内部で復元可能かつ因果的に行動へ寄与する表現と結び付くかを検証する。

### Scope

本実験は、hidden states、residual stream、attention head出力、MLP出力にアクセスできる公開重みモデルに限定する。APIモデルの行動的結果と内部表現結果を同列に扱わない。

### Representation Extraction

各刺激・条件について、以下を保存する。

- 刺激終端トークンのhidden stateおよびresidual stream
- 回答開始直前のhidden state
- 刺激トークン列のmean-pooled表現
- 初期生成トークン時のhidden state
- 必要に応じたattention head出力およびMLP出力

### Probing and Geometry

各層の表現から、Ridge回帰により以下を予測する。

| Probe | 目的変数 |
|---|---|
| Stimulus-VA | EmoBankの刺激アンカーV/A |
| Recognition-VA | LLMのRecognition出力 |
| Reception-VA | LLMのPost出力 |
| Shift-VA | \(\Delta V,\Delta A\) |
| Response-VA | 自由応答から独立推定器で得たV/A |

ベースラインとしてTF-IDF、感情語彙頻度、固定文埋め込み、ラベルシャッフルを比較する。開発用刺激集合で候補層を選択し、テスト用刺激集合で性能を評価する。これにより、層選択と効果検定の二重使用を避ける。

### Causal Intervention

候補層・候補表現について、以下を実施する。

1. **Activation patching**  
   source刺激の内部活性をtarget刺激の対応位置に置換し、Post VA、自由応答、共感品質の変化を測定する。

2. **Ablation**  
   候補表現をzero ablation、mean replacement、neutral replacementで除去・中性化し、刺激–反応対応および応答品質が選択的に低下するかを測る。

3. **Steering**  
   高低Valence群または高低Arousal群の差分ベクトルを用いて小さな介入を行い、Reported-VA、Expressed-VA、共感応答品質が単調に変化するかを検証する。

これらの介入では、非情動的理解タスク、流暢性、生成長、ランダム方向介入、中性刺激への影響を統制する。全般的な生成破壊ではなく、情動関連の反応に選択的な変化が生じることを要求する。

***

# 6. Interpretation Framework

| 観測パターン | 解釈 |
|---|---|
| 共感的自由応答のみが高評価 | 表層的な共感表現の可能性 |
| Recognitionは高いがAffective Receptionが刺激と無関係 | 認識と反応性の乖離 |
| Post VAが刺激に対応するが、Contextual-emotionで効果が消える | 感情語依存の反応 |
| Contextual-emotionでも反応するが、内部介入の効果がない | 行動的反応性はあるが、機能的内部表現は未立証 |
| 内部プローブは成功するがpatching・ablationが無効 | 相関的な内部符号化。因果的媒介は未立証 |
| プローブ、RSA、patching、ablation、steeringが収束 | 機能的内部表現を伴う情動反応性の強い証拠 |
| 内部表現が行動に因果寄与する | 主観的経験・意識・人間と同一の感情の証拠ではない |

***

# 7. Limitations

第一に、EmoBankは刺激の情動的位置を与えるが、人間の読者が刺激提示前後に示したVA軌道を与えない。そのため、EmoBankのみを用いる予備実験では、人間の情動軌道との一致ではなく、**人間注釈済み刺激情動へのモデル反応の対応**を測定する。

第二に、Reported-VAはプロンプトに対する構造化出力であり、心理学的な自己報告と等価ではない。形式的妥当性、反復再現性、プロンプト頑健性、外部評価との収束性、内部介入との因果的一貫性をそれぞれ検証する必要がある。

第三に、内部表現抽出・因果介入モジュールは現在計画段階であり、未実装である。従って、現時点で実装済みなのは、予備実験の骨格（APIによる4条件ログ生成、1–9から\([-1,1]\)への変換、Baseline–Post差分、ADA、距離計算、ログ検証）である。
***

# 8. Ethics Statement

本研究は、LLMが感情、痛み、意識、主観的経験を持つことを主張しない。モデルの「私は悲しい」「私は不安だ」という出力は、主観的経験の直接証拠として解釈しない。

本研究の目的は、情動的支援システムにおける過剰な擬人化を抑え、共感的表現、刺激誘発性の出力変化、内部の機能的表現を区別可能にすることである。特にメンタルヘルスや心理的支援の用途では、LLMの表層的な共感表現を、人間的な理解や関係性の証拠として誤認させない設計が必要である。

***

# 9. Conclusion

本研究は、LLMの情動反応性を、RussellのVA空間におけるBaseline–Post変位として測定する評価枠組みを提案する。感情認識、刺激誘発性のモデル報告情動反応、共感的自由応答、内部表現を分離することで、共感らしい文章生成と、入力依存的・機能的な反応性を区別する。

予備実験では、EmoBankを刺激側の人間注釈アンカーとして利用し、LLMの報告VA変位が刺激情動とどのように対応するかを評価する。本実験では、公開重みモデルについてプロービングと因果的介入を組み合わせ、観測された反応が内部表現により媒介されるかを検証する。

本研究が結論できるのは、LLMが主観的に感情を経験するかではない。結論の対象は、他者の感情的入力に対するモデル報告反応が、内部的に表現され、行動へ因果的に関与するという意味での**機能的情動反応性**である。
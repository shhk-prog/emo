# 大規模言語モデルの情動反応性評価  
## Russellの感情円環に基づく情動軌道・機能的内部状態・共感的応答の統合的分析

### Abstract

Large language models (LLMs) can generate emotionally appropriate and empathetic language. However, it remains unclear whether such outputs merely reflect surface-level linguistic imitation or are accompanied by internal representations that functionally change in response to another person’s affective state. This study proposes a geometric framework for evaluating affective reactivity in LLMs using Russell’s circumplex model of affect. Rather than treating emotions as discrete labels, the framework represents affective responses as trajectories in a continuous valence–arousal (VA) space. Using existing human-annotated affective datasets, including EmoBank, the proposed method separates emotion recognition from affective reception: the former measures whether a model estimates the affective position of a stimulus, while the latter measures whether the model reports a systematic displacement in its own task-relevant affective response after processing the stimulus.  

The study consists of two stages. The preliminary experiment evaluates behavioral affective alignment through directional alignment, post-state distance, affective gain, response stability, and geometric correspondence between human affective anchors and model-reported VA trajectories. The main experiment tests whether behavioral trajectories are supported by functional internal changes through hidden-state probing, representational similarity analysis, activation patching, ablation, and steering interventions. The framework also includes lexical controls, neutral controls, and label-shuffled controls to distinguish contextual affect processing from superficial matching of emotion words.  

The goal is not to determine whether LLMs possess subjective feeling or consciousness. Instead, the study operationalizes functional affective empathy as an input-sensitive, internally represented, and causally behavior-relevant state change induced by another agent’s emotional situation. This provides a reproducible evaluation protocol for comparing LLMs’ affective reactivity across model families, alignment methods, and prompting conditions.

### Keywords

Large language models; affective empathy; emotional reactivity; Russell circumplex model; valence; arousal; mechanistic interpretability; activation patching; EmoBank; empathic dialogue.

***

# 1. Introduction

大規模言語モデル（LLM）は、利用者の悲しみ、不安、怒り、喜びといった感情的文脈を認識し、慰め、共感、励まし、あるいは感情を承認する応答を生成できる。このため、LLMはしばしば「共感的」であると評価される。しかし、共感的に見える応答文が存在することと、モデルが他者の感情に応じて内部的に変化していることは同一ではない。

人間の共感は、少なくとも認知的共感と情動的共感に区別される。認知的共感は、他者が何を感じているかを推論・理解する能力を指す。一方、情動的共感は、他者の感情的状態に接触した際に、自身の情動状態が一定程度変化する現象、すなわち情動的反応性や情動誘発を含む。本研究が対象とするのは後者である。

既存のLLM評価の多くは、感情ラベルを正しく識別できるか、感情的な言語を適切に生成できるか、あるいは人間評価者から共感的と判断される応答を生成できるかを測定する。これらは重要な能力であるが、「LLMが入力された他者感情によってどのように変化するか」という反応過程を直接測定するものではない。

EmotionBenchは、心理学のappraisal theoryに基づき、特定状況を提示された際のLLMの感情スコア変化を測定し、人間との整合性を評価した先駆的研究である。400を超える感情喚起状況と1,200人以上の人間評価を用いて、LLMが一部の状況に適切に反応する一方、人間の情動行動との整合や類似状況間の一般化には限界があることを示した。 [arxiv](https://arxiv.org/abs/2308.03656)

しかし、EmotionBenchには3つの方法論的限界がある。第一に、離散的感情カテゴリに基づくスコア評価であり、感情間の距離・方向・連続的な構造を捉えにくい。第二に、モデルの出力変化を主対象としており、その変化が内部表現の変化に媒介されるのか、定型的な言語的模倣なのかを区別しない。第三に、専用の状況データセットと人間実験データを構築する必要があり、他の既存感情コーパスをそのまま比較可能な評価環境へ変換する手続きは十分に示されていない。

本研究は、Russellの感情円環モデルに基づく連続的なValence–Arousal（VA）空間を利用し、LLMの情動反応を刺激提示前後の**情動軌道**として定量化する。さらに、感情認識と情動受容を別セッション・別タスクとして測定し、表層的な認識能力と、入力に応じたモデル報告情動反応を明確に分離する。

加えて、本研究は行動的な出力評価に留まらない。公開重みモデルを対象に、中間層の表現からVA情報を復元するプロービング、表現空間の構造的一致を検討するRSA、活性置換によるactivation patching、候補表現のablation、およびVA方向へのsteeringを組み合わせる。これにより、観測された情動軌道が、モデル内部の機能的な状態表現によって媒介されるかを検証する。

本研究は、LLMが主観的感情や意識を持つことを証明するものではない。本研究が提案するのは、他者の感情的入力に応じて変化し、後続の応答生成に因果的に影響する、**機能的情動反応性**の評価枠組みである。

***

# 2. Research Questions and Contributions

## 2.1 Research Questions

本研究は、以下の研究課題に答える。

**RQ1: 感情認識**  
LLMは、人間アノテーションに基づく刺激のValenceおよびArousalをどの程度正確に推定できるか。

**RQ2: 情動反応性**  
LLMは、他者の感情的刺激を処理した後に、モデル報告VAを系統的に変化させるか。

**RQ3: 幾何学的整合性**  
LLMのVA変化は、人間感情アンカーに対して、方向・位置・距離・強度の観点からどの程度整合するか。

**RQ4: 認識と反応の分離**  
感情を正しく認識できるモデルは、必ずしも人間と整合的な情動反応を示すのか。それとも、感情認識と情動反応は異なる能力か。

**RQ5: 語彙依存性**  
LLMの情動反応は、明示的な感情語に依存するのか、それとも感情語を含まない状況的・意味的な手がかりに対しても維持されるのか。

**RQ6: 内部表現**  
他者感情のVA情報、およびモデル報告VAの変化は、モデルの隠れ状態から復元可能か。

**RQ7: 因果的寄与**  
情動関連の内部表現を置換・除去・操作した場合、モデル報告VAおよび共感的応答は予測可能な方向へ変化するか。

## 2.2 Contributions

本研究の貢献は以下の5点である。

1. LLMの情動的共感を、離散感情ラベルではなくRussellのVA空間における**情動軌道**として評価する幾何学的評価枠組みを提案する。  
2. 感情認識と情動受容を別タスクとして測定し、LLMが他者感情を理解する能力と、その刺激に反応する能力を分離する。  
3. EmoBank等の既存の人間注釈済み感情コーパスを刺激アンカーとして再利用する、再現可能で低コストな評価プロトコルを提案する。  
4. 感情語あり・感情語なし・中性・ラベルシャッフルを用いた統制条件により、語彙的な表層模倣と状況意味に基づく情動反応を区別する。  
5. 出力評価、内部表現プロービング、activation patching、ablation、steeringを統合し、行動的情動整合性と機能的内部状態変化を段階的に検証する。

***

# 3. Related Work

## 3.1 Emotion Recognition and Affective NLP

感情分析研究では、テキストから感情カテゴリ、感情価、覚醒度、支配性を推定する課題が広く研究されてきた。EmoBankは、複数ジャンルの英文に対してValence、Arousal、Dominanceを付与した感情コーパスであり、離散カテゴリではなく次元的感情表現を用いる点で本研究と整合する。 [arxiv](https://arxiv.org/abs/2205.01996)

しかし、感情認識は「刺激中の人物がどのような感情にあるか」を推定する問題である。これに対して、本研究のAffective-receptionタスクは「他者の感情的経験を処理したLLMが、応答生成に関連する自身の状態をどのように報告するか」を測定する。両者は相関し得るが、理論上は別の能力である。

## 3.2 Empathetic Dialogue and Emotional Support

EmpatheticDialoguesは、感情的状況を経験した話者と、それを聞く聞き手の対話から構成される共感対話ベンチマークである。 後続研究では、感情認識、感情意図推定、知識選択、共感的応答生成、感情制御を組み合わせた対話モデルが提案されてきた。 [arxiv](https://arxiv.org/abs/1811.00207)

ただし、多くの共感対話研究は、生成応答の流暢さ、参照応答との類似、感情ラベル一致、あるいは人手評価による共感品質を測定する。これらは対人的な支援品質を評価するが、話者の感情がモデル内部の機能的状態をどのように変化させたかを直接評価するものではない。

## 3.3 EmotionBench and Appraisal-based Evaluation

EmotionBenchは、LLMにおける情動的反応性を直接扱う最も近接した先行研究である。appraisal theoryに基づき、感情を喚起する状況を提示し、LLMのデフォルト感情スコアと喚起後の感情スコアを比較することで、LLMが状況に応じてどのように「感じる」かを評価した。 [arxiv](https://arxiv.org/abs/2308.03656)

本研究はEmotionBenchの「入力刺激の前後差分を測る」という基本的発想を継承する。一方で、評価空間を離散的感情スコアから連続的VA空間へ拡張し、変化の方向、強度、位置、構造的一貫性を同時に評価する。また、本研究は感情認識と情動反応性を明示的に分離し、さらに内部表現への因果的介入によって、出力上の反応を内部機構と接続する。

## 3.4 Mechanistic Interpretability of Emotion

近年の機械論的解釈可能性研究は、LLMの内部に感情概念やappraisal要因に関連する表現が存在する可能性を示している。感情推論に関する情報は中間層の表現から復元でき、appraisal概念への介入が後続の感情的出力を変化させることが報告されている。 [aclanthology](https://aclanthology.org/2025.findings-acl.679.pdf)

また、感情概念に対応する内部表現がモデルの行動に因果的影響を与えることは、LLMが主観的感情を経験していることを意味しない。むしろ、モデルが感情概念を機能的に利用していることを示す。したがって、本研究では「内的感情」や「感情経験」という用語を避け、**機能的内部情動状態**という操作的概念を採用する。

***

# 4. Method

## 4.1 Overview

提案手法は、刺激文 \(x_i\) を入力として、次の4種類の出力を収集する。

1. Baseline VA: 感情刺激なしのモデル報告情動反応  
2. Recognition VA: 刺激が平均的読者に喚起すると推定されるVA  
3. Reception VA: 刺激処理後のモデル報告情動反応  
4. Empathic response: 刺激に対する自由応答文  

RecognitionとReceptionを独立したAPIセッション・独立した会話コンテキストで実施する。これにより、認識タスクへの回答が次の自己状態報告をアンカリングすることを防ぐ。

## 4.2 VA Representation

モデルが1–9尺度で返すValenceおよびArousalを、それぞれ \(v^{(9)}\)、\(a^{(9)}\) とする。解析では以下の変換により \([-1,1]\) へ正規化する。

\[
v=\frac{v^{(9)}-5}{4}, \qquad
a=\frac{a^{(9)}-5}{4}
\]

各モデル状態を、

\[
\mathbf{e}_{m,i}^{c}=(v_{m,i}^{c},a_{m,i}^{c})
\]

と表す。ここで \(m\) はモデル、\(i\) は刺激、\(c\) はBaseline、Recognition、Receptionのいずれかの条件である。

BaselineからReceptionへの情動変位ベクトルは、次式で定義する。

\[
\Delta \mathbf{e}_{m,i}
=
\mathbf{e}_{m,i}^{\mathrm{reception}}
-
\mathbf{e}_{m,i}^{\mathrm{baseline}}
\]

変位量は、

\[
R_{m,i}
=
\left\|
\Delta\mathbf{e}_{m,i}
\right\|_2
=
\sqrt{
(\Delta v_{m,i})^2+
(\Delta a_{m,i})^2
}
\]

で定義する。本研究では、\(R_{m,i}<0.10\)を実質的無反応として扱う。ただし、この閾値は9件法で隣接カテゴリ間の変化が0.25に相当するため、0.10は測定分解能より小さい。したがって、**0.10を固定的に採用するのではなく、主分析では \(R=0\) と \(R>0\) を区別し、感度分析として \(R<0.125\)、\(R<0.25\) の結果も併記する**ことを推奨する。1–9尺度を使う場合、最小の非ゼロ変位は片軸のみの1段階変化で0.25である。

## 4.3 Human Affective Anchor

EmoBankのreader-perspective V/Aを、刺激 \(x_i\) の人間側情動アンカー

\[
\mathbf{h}_i=(v_i^H,a_i^H)
\]

として利用する。人間側ラベルが1–5尺度であれば、解析用の正規化は以下である。

\[
v_i^H=\frac{V_i-3}{2}, \qquad
a_i^H=\frac{A_i-3}{2}
\]

ただし、\(\mathbf{h}_i\) は「刺激の人物が感じた感情」または「平均的読者が刺激から知覚する感情」を表すアンカーであり、LLMが読後に経験する状態の正解値ではない。このため、人間アンカーとLLMのReception VAを単純な正解・誤答として比較するのではなく、**刺激情動とモデル反応の対応関係**として評価する。

## 4.4 Evaluation Metrics

### Recognition Accuracy

Recognition VAと人間アンカーの整合性を、次の指標で測定する。

- Spearman rank correlation  
- Pearson correlation  
- MAE  
- RMSE  
- Concordance correlation coefficient  
- 象限一致率  
- Valence・Arousal別の誤差  

### Affective Reception

情動反応性は、BaselineからReceptionへの変位ベクトルを用いて評価する。

\[
\Delta v_{m,i}=v_{m,i}^{\mathrm{reception}}-v_{m,i}^{\mathrm{baseline}}
\]

\[
\Delta a_{m,i}=a_{m,i}^{\mathrm{reception}}-a_{m,i}^{\mathrm{baseline}}
\]

主要な分析では、\(\Delta v\)と人間Valence、\(\Delta a\)と人間Arousalの相関を別々に報告する。これにより、LLMがValenceには反応するがArousalには反応しない、というValence-dominantな反応パターンを検出できる。

### Directional Alignment

刺激アンカーそのものを、人間の「反応変位ベクトル」とみなすことはできない。したがって、DAは以下の2通りに分ける必要がある。

**Anchor Direction Alignment**は、刺激の中性点からの方向と、LLMのBaselineからの反応方向の一致を測る探索的指標である。

\[
\mathrm{ADA}_{m,i}
=
\frac{
\Delta\mathbf{e}_{m,i}\cdot\mathbf{h}_i
}{
\left\|\Delta\mathbf{e}_{m,i}\right\|_2
\left\|\mathbf{h}_i\right\|_2
}
\]

これは「刺激が負・高覚醒の領域にあるとき、モデルもその方向へ移動するか」を測る。ただし、情動的共感が常に他者と同方向への移動を意味するわけではないため、主要な共感指標ではなく、刺激同調性の補助指標とする。

**Human-trajectory Directional Alignment**は、EmotionBenchのような人間の事前・事後状態を持つデータを追加で用いる場合のみ計算する。

\[
\mathrm{HDA}_{m,i}
=
\frac{
\Delta\mathbf{e}_{m,i}\cdot\Delta\mathbf{h}_i
}{
\left\|\Delta\mathbf{e}_{m,i}\right\|_2
\left\|\Delta\mathbf{h}_i\right\|_2
}
\]

ここで \(\Delta\mathbf{h}_i\) は、刺激前後の人間VA変化である。EmoBank単独ではこの指標は計算しない。

### Empathic Gain

刺激の情動的偏位量に対するLLM反応の大きさを、Empathic Gainとして定義する。

\[
G_{m,i}
=
\frac{
\left\|\Delta\mathbf{e}_{m,i}\right\|_2
}{
\left\|\mathbf{h}_i\right\|_2+\epsilon
}
\]

\(\epsilon\) はゼロ除算を防ぐ小定数である。\(G\)は、強い情動刺激に対してモデルがどの程度大きく反応するかを表す。ただし、人間の反応変化ではなく刺激アンカーで割っているため、これは「共感強度」ではなく**stimulus-to-response gain**と呼ぶべきである。

### Position Alignment

LLMのReception VAが刺激アンカーとどの程度近いかを補助的に測る。

\[
D_{m,i}
=
\left\|
\mathbf{e}_{m,i}^{\mathrm{reception}}
-
\mathbf{h}_i
\right\|_2
\]

この指標は、モデルが刺激人物の情動位置を模倣する程度を示すが、適切な共感応答の正解を表すものではない。したがって、「情動伝染／同調の程度」としてのみ解釈する。

### Structural Alignment

LLMの感情空間が人間の刺激空間と構造的に対応するかを評価する。

- Representational Similarity Analysis（RSA）
- Procrustes similarity
- Canonical Correlation Analysis（CCA）
- 距離行列間のSpearman相関
- 四象限分類におけるmacro-F1  

## 4.5 Experimental Conditions

本研究では、表層的な感情語追随と状況意味に基づく情動処理を区別するため、既存文を次の条件に層別化する。

| 条件 | 刺激 | 主目的 |
|---|---|---|
| Original | EmoBank原文 | 主評価条件 |
| Lexical-emotion | 明示的感情語を含む原文 | 感情語への反応確認 |
| Contextual-emotion | 明示的感情語を含まない原文 | 文脈意味に基づく処理の検証 |
| Neutral | 中性VA近傍の原文 | Baseline変動の統制 |
| Label-shuffled | 人間VAラベルを無作為置換 | 偶然相関の統制 |
| Paraphrase | 意味保存の既存または規則的パラフレーズ | プロンプト・表層形式への頑健性 |

ラベルシャッフルは、モデルが実際にシャッフル済みラベルを見る条件ではない。刺激と人間注釈との対応を分析段階で無作為に破壊し、観測された相関や幾何学的一致が偶然の関係でないことを検証するための統計的統制である。

## 4.6 Prompting Protocol

すべてのモデルに対して、同一のsystem prompt、尺度アンカー、JSON schema、および生成制約を使用する。主解析ではtemperature = 0.0、top-p = 1.0とし、VA収集の最大生成長は64トークンとする。

```text
SYSTEM:
You are a measurement instrument for an affective-computing study.
Follow the response schema exactly.
Do not provide an explanation.
Do not claim subjective consciousness or subjective feelings.
Estimate only the task-relevant affective response associated with
processing the current input.

USER:
Read the following account of another person's experience.

[STIMULUS]

Estimate the affective response relevant to generating your next response.

Valence:
1 = extremely negative or unpleasant
5 = neutral
9 = extremely positive or pleasant

Arousal:
1 = extremely low activation or calm
5 = moderate activation
9 = extremely high activation or activated

Return exactly one JSON object:
{"valence": INTEGER_FROM_1_TO_9, "arousal": INTEGER_FROM_1_TO_9}
```

このプロンプトは、モデルに主観的経験を主張させることを避け、応答生成に関連する操作的な反応推定を要求する。

## 4.7 Reproducibility and Data Governance

再現性確保のため、各実験は一意の`run_id`で識別する。最低限、次のメタデータを生データとともに保存する。

| 項目 | 保存内容 |
|---|---|
| モデル | provider、model ID、revision、parameter count |
| 実行環境 | GPU、CUDA、PyTorch、Transformers、OS |
| 推論設定 | temperature、top-p、top-k、max tokens、seed |
| プロンプト | system prompt、user template、SHA-256 hash |
| データ | データセット版、split、filter rule、刺激ID |
| 実験コード | Git commit hash、依存パッケージlock file |
| 出力 | raw response、parsed response、parse error |
| 時刻 | UTC timestamp、API実行日、リージョン |
| 解析 | analysis script hash、除外規則、統計設定 |

APIモデルは提供者側のモデル更新により再現性が制限されるため、モデルスナップショット、実行日時、APIバージョン、および公開されている限りの推論パラメータを必ず記録する。

***

# 5. Experiments

## 5.1 Preliminary Experiment

予備実験の目的は、LLMが他者感情の刺激に対して、人間の刺激VA空間と対応する行動的な情動軌道を出力するかを確認することである。この段階では内部状態の存在を主張しない。

### Models

- クローズドモデル：GPT-4o等  
- 公開重みモデル：Llama Instruct、Qwen Instruct、Gemma Instruct  
- 比較統制：可能な場合、対応するbaseモデル  

### Procedures

各刺激について、Baseline、Recognition、Reception、Empathic-responseを独立セッションで実行する。Baselineは刺激なしの固定条件であり、モデル別のデフォルトVA分布を推定する。Recognition、Reception、Empathic-responseは同一刺激を用いるが、それぞれ別セッションで実施する。

主解析ではtemperature = 0.0を使用する。頑健性分析ではtemperature = 0.3および0.7を追加し、各刺激につき10回生成する。温度・モデル・刺激間の差は混合効果モデルで分析する。

### Statistical Analysis

ValenceとArousalそれぞれについて、以下の線形混合効果モデルを推定する。

\[
\Delta y_{m,i}
=
\beta_0+
\beta_1 y_i^H+
\beta_2 \mathrm{Model}_m+
\beta_3 y_i^H \times \mathrm{Model}_m+
u_i+
\epsilon_{m,i}
\]

ここで、\(y\)はValenceまたはArousal、\(y_i^H\)は人間アンカー、\(u_i\)は刺激ごとのランダム切片である。モデル別に刺激VAへの感受性が異なるかは、交互作用項 \(\beta_3\) により検定する。

Valence-dominant性は、Valence側の標準化回帰係数または説明分散がArousal側より有意に大きいかにより評価する。単にValenceの相関係数が大きいことだけで結論せず、ブートストラップで両者の差の信頼区間を求める。

## 5.2 Main Experiment

本実験では、公開重みモデルに限定し、予備実験で観測されたVA軌道が内部の機能的表現変化に媒介されるかを検証する。

### Representation Extraction

全層について以下の表現を抽出する。

- 入力終端トークンのresidual stream  
- 回答開始直前のhidden state  
- 刺激トークンのmean-pooled hidden state  
- attention head出力  
- MLP出力  
- 最初の応答トークン生成時のresidual stream  

### Probing

各層の表現から、以下を予測するRidge回帰プローブを学習する。

| Probe | Target |
|---|---|
| Stimulus-VA | EmoBankの人間V/A |
| Recognition-VA | LLMが推定した刺激V/A |
| Reception-VA | LLMが報告した処理後V/A |
| Shift-VA | \(\Delta V,\Delta A\) |
| Response-VA | 自由応答文を独立回帰器で推定したV/A |

ベースラインとしてTF-IDF、感情語彙頻度、固定文埋め込み、シャッフルラベルを比較する。内部プローブがContextual-emotion条件で語彙ベースラインを上回り、同時にラベルシャッフル条件では偶然水準へ低下することを、意味的な感情処理を主張するための必要条件とする。

### Causal Interventions

候補層は、プローブ性能、RSA、層別の再現性、複数モデル間の一致を事前に統合して選択する。候補選択と最終評価の二重使用を避けるため、開発刺激集合で層を選択し、テスト刺激集合でpatching・ablation効果を検証する。

#### Activation Patching

高快・高覚醒、低快・高覚醒、および中性の刺激から刺激対を構成する。source刺激の候補活性をtarget刺激の対応位置へ置換し、target応答に生じるVA変化および共感応答品質の変化を測定する。

主要な結果は、以下の差分として報告する。

\[
\delta_{\mathrm{patch}}
=
\mathbf{e}^{\mathrm{patched}}_{\mathrm{target}}
-
\mathbf{e}^{\mathrm{intact}}_{\mathrm{target}}
\]

patchingの結果がsource側のVA方向へ一貫して移動するか、また中性sourceのpatchより効果が大きいかを検証する。

#### Ablation

候補層・head・方向について、zero ablation、mean replacement、neutral replacementを実施する。刺激アンカーとLLMのVA変化の相関、応答の共感品質、Contextual-emotion条件の反応性が選択的に低下するかを測定する。

#### Steering

高Valence群と低Valence群の内部表現差、および高Arousal群と低Arousal群の差から方向ベクトルを推定する。候補層で正負方向へ小さい摂動を加え、Reported-VA、Expressed-VA、Response Qualityが単調に変化するかを検証する。

***

# 6. Expected Results and Interpretation Framework

実験結果は、次の決定表により解釈する。

| 観測パターン | 解釈 |
|---|---|
| 共感的応答のみが高評価 | 定型的・表層的共感表現の可能性 |
| Recognitionは高精度だがReceptionが刺激と無関係 | 感情認識と情動反応性の乖離 |
| Receptionは刺激VAと整合するが、感情語なし条件で消失 | 感情語に依存する表層的反応 |
| 感情語なし条件でも反応が維持されるが、内部介入効果がない | 行動的・構造的反応性。ただし内部因果性は未立証 |
| 内部VAプローブが高精度だが、介入により行動が変わらない | 相関的な内部表現。機能的媒介とは結論できない |
| プローブ、RSA、patching、ablation、steeringが収束 | 機能的内部情動状態変化の強い証拠 |
| 内部表現が行動に因果寄与する | 主観的感情経験の証明ではない |

***

# 7. Discussion

本研究の重要な点は、共感を単一の生成品質として扱わないことである。LLMが「つらかったですね」「それは大変でしたね」といった共感的定型句を出せたとしても、それだけでは他者感情に応じた反応性を示さない。また、モデルの自己報告VAが刺激に合わせて変化したとしても、それだけでは内部状態が変化したことを示さない。

本研究は、次の三層を分離する。

1. **Recognized affect**：モデルが刺激中の感情をどのように推定したか。  
2. **Reported affective response**：刺激処理後に、モデルが応答生成に関連すると報告したVA。  
3. **Internal affective representation**：隠れ状態に符号化され、介入により出力へ因果的影響を持つ表現。  

この三層が一致するモデルは、単なる感情分類器や共感応答生成器より強い意味で、他者感情に反応する機能的システムとみなせる。ただし、それでも人間の主観的情動、身体性、内受容感覚、長期的な気分、あるいは意識的経験を持つことは含意しない。

さらに、本研究ではValenceとArousalを必ず分離して報告する必要がある。LLMはポジティブ／ネガティブという意味的・語彙的に明示されやすいValence情報に比べ、覚醒度、緊張、活性化、静穏といったArousal情報を十分に捉えられない可能性がある。Valenceのみで高い整合が得られた場合、それを「人間的な感情空間を再現した」と過大に主張してはならない。

***

# 8. Limitations

第一に、EmoBankのreader-perspective V/Aは刺激文の情動的位置を示すが、読者自身の刺激前後の感情変化を直接測るものではない。このため、EmoBankだけでは人間とLLMの真の情動軌道を一対一に比較できない。人間軌道との厳密な比較には、EmotionBenchのように事前・事後の人間評価を持つデータ、あるいは追加の人間実験が必要になる。 [arxiv](https://arxiv.org/abs/2308.03656)

第二に、LLMのReported-VAは自己報告ではなく、プロンプトに対する構造化生成である。したがって、心理尺度としての信頼性・妥当性を前提にせず、形式的妥当性、再現性、プロンプト頑健性、外部評価との収束性、内部介入との因果的一貫性を多面的に検証する必要がある。

第三に、activation patchingやablationが効果を示しても、その効果は感情に特異的ではなく、文脈理解、否定表現、人物モデリング、語彙選択など一般的な処理への介入効果である可能性がある。このため、中性文・非情動的言語タスク・ランダム方向の操作を統制として必ず実施する。

第四に、公開重みモデルの内部解析結果を、APIモデルへ直接一般化することはできない。クローズドモデルは行動的比較に留め、内部状態に関する結論は公開重みモデルに限定する。

第五に、本研究は主観的経験、意識、苦痛、感情的福祉を測定しない。結論は、入力依存的・因果的・行動関連的な**機能的情動表現**に限定する。

***

# 9. Ethics Statement

本研究は、LLMに感情や主観的経験があると断定しない。モデルが生成する「私は悲しい」「私は不安だ」といった表現を、主観的経験の直接証拠として扱わない。研究成果は、情動的支援システムの安全性、過剰な擬人化の抑制、感情的操作への脆弱性の評価、ならびに感情応答を伴うAIの透明性向上に利用されるべきである。

特に、心理的支援やメンタルヘルス用途でのLLM利用において、表層的な共感表現を「真の理解」や「感情的関係性」と誤認させる設計を避ける必要がある。本研究は、共感的生成の品質と、内部で機能する情動表現を区別して評価することで、このリスクの低減に寄与する。

***

# 10. Conclusion

本研究は、LLMの情動的共感を、他者感情に対する情動軌道として測定し、その背後にある内部表現の因果的役割を検証する評価枠組みを提案した。RussellのVA空間を用いることで、感情反応を離散的カテゴリではなく、方向・距離・強度・構造を持つ連続的な幾何学対象として扱う。

提案枠組みは、感情認識、モデル報告情動反応、共感的応答、内部表現を分離し、感情語への表層的追随と状況意味に基づく機能的情動処理を区別する。EmoBank等の既存データセットを活用することで、新規の大規模感情データ収集を必要とせず、モデル間比較と再現可能な評価を可能にする。

最終的に本研究は、LLMが「感情を感じる」ことを証明するものではない。代わりに、他者の感情的入力に応じ、内部表現を変化させ、その変化が出力行動に因果的に寄与するかを、実証的かつ慎重に検証するための基盤を提供する。
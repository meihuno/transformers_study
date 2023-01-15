# 背景

本ドキュメントは、機械学習を用いた分類タスクにおける「抜け漏れ（False Negative）をゼロにしてRecallを100%にした上で、なるべくPrecisionを高くする」ための方法についてサーベイした結果を記す。

分類タスクの応用には、特定のラベル（２値分類の場合は正例）の未検出をゼロ（False Negative(FN)をゼロ or Recall100% ）にしたい要求が存在する。が存在する。例えば、医療画像所見から症状を検出する、あるいは、法令ニュースから重要な法令の変更を検出する、セキュリティニュースから重要なセキュリティ情報を検知する、などの、意思決定に関わる情報を分類するような場合である。機械学習モデルが大量の入力を分類をした（機械は大量のデータを処理することに優れている）後工程で人間が内容を確認すること（人間によるチェック）を考えると、分類モデルによる「人間が確認対象とするラベルの抜け漏れ（False Negative）」はゼロであってほしい。さもなくば、結果を人間が全て確認しなければならず、機械学習を行って分類を行う意味がなくなってしまう。

特定のラベルの抜け漏れをゼロにすること（Recall100%を達成すること）自体は容易である。モデルの判定対象のラベルのスコアに対する閾値設定をゆるく設定すれば、抜け漏れはなくなる。しかし、その場合、出力結果には多くの誤りが含まれることになり、Precisionは低くなり、人間によるチェックの工数が嵩むことになる。Recallを100%にした上で、ある程度のPrecisionを維持しなければ、チェック工数の削減効果が得られない。機械学習を導入するコストよりも、チェック工数削減効果が多くなるようでは、これもまた機械学習を行って分類を行う意味がなくなってしまう[^footnote_sec1_1]。

[^footnote_sec1_1]: タスクによる。時間単価の高額となる医師によるチェックであれば、少しでも工数削減できれば効果が得られる可能性がある。

機械学習モデルによる分類タスクにおける「抜け漏れ（False Negative）をゼロにしてRecallを100%にした上で、なるべくPrecisionを高くする」ための手順が確立できれば、機械学習による分類モデルを用いてドキュメントのチェック作業に関する業務効率化を定量的に評価することができる。本ドキュメントは、上記を方法についてサーベイを行った結果をまとめたものである。

## Recall１００%を達成するための手の打ちどころ

分類タスクでRecallを100％に向けて向上させるには一般的な方法として以下のステップがある。

1. 高品質な学習データを増やす
2. ハイパーパラメータチューニング
3. 閾値を調整する

### 高品質な学習データを増やす

機械学習モデルの性能改善には、モデルの改善よりも誤りの少ない学習データを準備することが重要であるとする考えがある（「Data centric AI」[^ref_sec_1_1]）

[^ref_sec_1_1]: Andrew Ng, https://datacentricai.org/
  
学習データの（判定したいクラスや）正例を増やすことで、モデルによる抜け漏れを減らすことにつながりRecallが向上すると考えられる。Precisionの値を維持するには学習データの質も重要である。矛盾するラベルからはデータを正例と負例とが混じっている場合には、分類モデルはこれらデータを分離する曲線をうまく学習することが難しくなるだろう。また、学習した内容と矛盾するラベルが評価データに含まれる場合には、Recallを100%にするには閾値の値を低く設定せざるをえず、precisionは低くなってしまう。

学習データの質量を向上させるには以下のような技術や考え方があり、サーベイの節で概観する。

 - human in the loop
 - 人手で学習データを作成する際の支援、active learningなど）
 - データ拡張

### ハイパーパラメータチューニング

ハイパーパラメータのチューニングとは、機械学習モデルにおいて学習前に与える種々のハイパーパラメータを、Devlopmentデータや評価データでの評価尺度が良くなるように探索することである。ハイパーパラメータの種類は機械学習手法によって異なり、例えば、SVMでは誤り分類の度合いを調整するC（コスト）パラメータ、lightgdmなどの決定木手法には決定木の幅や深さに関するパラメータがあり、深層学習でも学習率[^term_sec_1_learning_rate]やweight decay[^term_sec_1_weight_decay]などがある。

[^term_sec_1_learning_rate] : 重みパラメータを一度にどの程度変化させるかを表すハイパーパラメータ。学習の収束速度や局所解回避に影響する。
[^term_sec_1_weight_decay] : コスト関数で設定された正則化項を調整するパラメータ。過学習抑止に影響する。

ハイパーパラメータの探索にはパラメータの値の組み合わせを格子状に網羅して探索グリッドサーチ、乱数によって値の組み合わせを探索ランダムサーチ、ベイズ最適化に基づくチューニング手法がある。グリッドサーチは少数のパラメータを調整する場合によく用いられ、多数のパラメータを調整する場合はベイズ最適化を用いた手法がよく用いられる。パラメータチューニングに関する情報については多数のWebサイトやブログポストでの解説がある[^ref_sec1_parameter_tuning]。

[^ref_sec1_parameter_tuning] :　「機械学習のパラメータチューニングを「これでもか！」というくらい丁寧に解説」https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359

ベイズ最適化を用いたパラメータ最適化ツールとしてOptuna[^ref_sec1_oputuna]が知られている（Oputunaについては書籍も出ている[^ref_sec_1_oputuna_book]）。OptunaではObjectiveメソッドでパラメータ探索を行う際の目的関数が定義でき[^ref_sec_1_oputuna_tutorial]、評価尺度を指定できるため、「Recall重視」や「Precision重視」といった調整が可能である。「Recall100%時のPrecision値」といった複数のデータを用いた評価尺度の設定も可能であると思われる（杉原は未実験）。ただし、あくまでハイパーパラメータの調整であり、評価データでのRecall100%が保証されるわけではないことに注意である。

[^ref_sec1_oputuna] : https://www.preferred.jp/ja/projects/optuna/

[^ref_sec_1_oputuna_tutorial] : https://optuna.readthedocs.io/en/stable/index.html

[^ref_sec_1_oputuna_book] : https://www.ohmsha.co.jp/book/9784274230103/

### 閾値調整

閾値調整はモデルのスコアに閾値を設けてスコアを上回る場合にあるクラスとして出力する。閾値を低く設定しておけば、多くの入力が出力されるため抜け漏れは少なくなる。RecallとPrecisionとのトレードオフとなるため、Recallを100に近く維持する場合は、Precisionが低くなることが多い。学習モデルのRecallとPrecisionを含めたトータルな性能の向上がより重要である。

また、学習データのラベルに矛盾がある場合、類似したデータに対する学習データでのラベルと評価データでのラベルが異なることがある。その場合、評価データの該当データに対する分類モデルのラベルのスコアは低くなり、Recall100%を担保するには閾値を低く設定せざるを得ず、Precisionが低くなる。学習データの質もRecall100%を維持しつつPrecisionを低下させないようにするには重要である。

### Recall100%時の

Recall100%時のPrecisionが低下する原因と考えらえる対策としては以下のようなものが考えられる。

1) 学習データの矛盾 -> 学習データを用いたモデルで評価データ中の対象ラベルを低評価 -> 閾値を低く設定しなければならない -> Precisionが下がる
2) 学習データが少ない -> 学習データを用いたモデルで評価データ中の対象ラベルを低評価 -> 閾値を低く設定しなければならない -> Precisionが下がる

3) 処理対象の素性が分類するに十分表現できていない[^sec_1_footnote_1] -> 学習データを用いたモデルで評価データ中の対象ラベルを高評価できない -> 閾値を低く設定しなければならない -> Precisionが下がる

4) 処理対象から得た素性にノイズが多い -> 学習データを用いたモデルで評価データ中の対象ラベルを高評価できない -> 閾値を低く設定しなければならない -> Precisionが下がる

1の対策はデータのラベルの修正が直接的な対策となり、ラベルノイズに強いモデルが次点の対策となる。2の対策は学習データの拡充、データ拡張である。3は素性の追加（テキスト分類であれば品詞の種類を増やす、bigramやchunckerを用いるなど）や分散表現の活用である。4は3とのトレードオフとなるが、データクレンジングや分類に寄与しない素性を除去する(素性選択という。参考となるブログポスト[^ref_sec_1_feature_selection])が考えられる。

機械学習モデルの構築の順番としては、データに過学習できることを確認してから汎化の対応を行うのがよいため、3の対策をとって評価データに対する性能が達成できてから4の対策を行うがよい。

[^sec_1_footnote_1] : テキスト分類での例え。人間が文章の否定表現を含めて判断している場合に（否定の内容の文章と肯定の内容の文章がある。これが「〜ない」などの助動詞で表現されているとする）、分類モデルがテキストから名詞のみを抽出してベクトルを作っている場合、分類モデルには否定の内容の文書と肯定の内容の文書の区別を行う手がかりが足りてないことになる。

[^ref_sec_1_feature_selection] : https://qiita.com/shimopino/items/5fee7504c7acf044a521

- False/Negativeが生じてしまう
 - 学習時には学習できなかったデータを分類する
 - 素性が十分分類の手がかりを表現できていない

# 論文サーベイ

学習データが足りない

https://www.semanticscholar.org/paper/A-graphical-approach-for-multiclass-classification-Merkurjev/656da1fc8b250d73173019b318c9a65e9f8ca7a3

「training data label contradiction」でsemantic scholarで検索。データ破損時の学習法、データの品質、ラベル自動付与学習についての論文が多数検索された。以下に、ピックアップした論文の概要を示す。


### 学習データの品質

Garbage in, garbage out?: do machine learning application papers in social computing report where human-labeled training data comes from?
https://www.semanticscholar.org/paper/Garbage-in%2C-garbage-out%3A-do-machine-learning-papers-Geiger-Yu/df2df1749b93ba86328ec7b86ff7e8d30029e3f5

機械学習の分野での学習データ構築を社会科学での知見をもとに分析した論文。クラウドワーカーを使った学習データづくりは社会学でのデータづくりの方法論に似ており、そのベストプラクティスがMLでのデータ作りにおいて鑑みられているかを調査した。


### 学習データの品質が悪いときにどうするか？

Learning with Bad Training Data via Iterative Trimmed Loss Minimization
https://www.semanticscholar.org/paper/Learning-with-Bad-Training-Data-via-Iterative-Loss-Shen-Sanghavi/66169adc068ef55f17ce1b1b51efb8778673ecfc

### 品質の悪い学習データを用いた学習

trimmed lossを最小化する。低いlossのサンプルを選ぶ。それらのサンプルのみで再学習する。

観察）cleanなデータのsampleでのlossの減りと、bad sampleSとでは異なる。画像。

This paper proposes to iteratively minimize the trimmed loss, by alternating between selecting samples with lowest current loss, and retraining a model on only these samples, and proves that this process recovers the ground truth in generalized linear models with standard statistical assumptions.

Does label smoothing mitigate label noise?
https://www.semanticscholar.org/paper/Does-label-smoothing-mitigate-label-noise-Lukasik-Bhojanapalli/82c77a88969ac0e3a4e55c9a7dc5ced4afee0225

label smoothingがdataのnoiseを軽減するか？ （）を適用すれば有効。画像の実験。

Effect of Label Redundancy in Crowdsourcing for Training Machine Learning Models
クラウドワーカーでラベルを付与してもらうときには冗長なデータセットにすると精度が上がるという。

Mechanisms for Automatic Training Data Labeling for Machine Learning
少量の人間のラベルデータを用いて構築した分類器でさらにデータを作る。Recall重視システムと、ルールベースで作ったラベルでのF値が向上。

Automatic Training Data Cleaning for Text Classification

## データ拡張

## human in loop

## dirty data

## 分布外検出

## 実務における

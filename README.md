# Susceptibility of Domain-Specific NLP Models to Adversarial Attack
## Introduction
Large language models such as BERT (Devlin et al. 2018) have been widely used for
domain-specific NLP tasks, whether it is extracting clinical diagnoses (Lee et al. 2020),
classifying financial documents (Yang et al. 2020), or detecting human emotions (Barbieri et al.
2020). Many of these models are popularly applied out-of-the-box to real-world scenarios - a
trend that is encouraged by the development of open-source machine-learning (ML) platforms,
such as Hugging Face[^1], that offer easy consumer access to models and enable rapid prototyping
of pipelines. However, the use of these language models comes with risks. The vulnerability of
BERT models to adversarial attacks has been widely documented (Jin et al. 2019; Dirkson et al.
2021). As the publication of most open-source models is seldom accompanied by an analysis of
model robustness, it is reasonable to be wary of their weaknesses when applying them in
specialized areas. This study conducts an empirical analysis of the robustness of these models to
demonstrate their weaknesses and to better understand how to guard against them. In doing so,
we highlight the need to overcome the fragility of these frameworks as they become more
integrated into our lives.

We perform a range of adversarial attacks on highly popular open-source BERT models to test
their robustness. We focus on two use cases that are typical examples of ML-enabled NLP
applications: Financial sentiment analysis and emotion classification from social media posts.
This empirical analysis aims to test and demonstrate the risk of using off-the-shelf machine
learning-based NLP tools for real-world applications. We compare and contrast a number of
attack paradigms in our empirical analysis to gain a more holistic understanding of the source of
weaknesses for each model. For each attack, we also compute a stealthiness score (GRUEN) that
evaluates the quality of the modified text. This is to ensure that corrupted sentences remain
comparable to the original inputs.

This paper has two contributions. One, we analyze Financial NLP models across two different
training strategies: The first pre-train model on domain-specific corpora (FinBert-tone[^2]) and the
second pre-train using a generic global corpus (FinBert-Prosus AI[^3]). In both instances, we
evaluate model performance when they are fine-tuned on the same dataset and task (sentiments
classification). Language models pre-trained on specific domains (such as biology-themed text) (Lee et al. 2020) or financial corpora (Yang et al. 2020)) often outperform BERT pre-trained on
generic corpora (eg. Wikipedia) in various NLP tasks, but the lower variance in training data
may result in models more susceptible to certain kinds of attacks. It is not obvious how the
choice of training strategy affects model robustness; our comparisons provide key insight into
their relationship.

Secondly, we also analyze the sensitivity of model robustness when we increase the size of
training data. While we expect that increase in training data increases both model robustness and
performance, it is unclear to what degree robustness depends on training data size. We hope to
estimate this through a comparison of two models trained on emotion classification in the social
media domain (Twitter). One model is pretrained on 850M English Tweets
(bertweet-base-emotion[^4]) and the other model (twitter-roberta-base-emotion[^5]) is pretrained on
60M tweets.

While a few BERT models, such as bioBERT (Lee et al. 2020) and SciBERT (Beltagy, Lo and
Cohan 2019) have been attacked in previous research, our approach focuses on sentiment
analysis from financial corpus and emotion classification of social media text. We choose these
domains as they are well-known examples of real-world NLP applications and they are amongst
the most popular open-source NLP models. The financial sentiment analysis models we test have
~700k and ~170k monthly downloads from the Hugging Face library - ranking 3rd and 11th of
all text classification models available. The models for emotions classification are also quite
popular, with up to 55k monthly downloads.

## Related Work
### Adversarial attacks
Adversarial attacks are intentional attempts to trick models into outputting the incorrect value by
providing carefully crafted input samples known as adversarial examples. Attacks can be broadly
classified into black-box and white-box methods, where the former does not require access to
model parameters to generate adversarial examples. Examples of black-box attacks are noise
adversaries, which emulate spelling errors committed by human beings, and synonymy
adversaries, which test if a model can understand synonymy relations (Goodfellow et al. 2014;
Yao, Qi 2021). White-box approaches exploit knowledge of the model parameters by modifying
input in the direction of the gradient of the loss function (Garg and Ramakrishnan, 2020). The
black-box attacks are more realistic in industrial or commercial settings. However, white-box
attacks allow us to test the robustness of the models and uncover architectural vulnerabilities. In
this experiment, we will apply both types of attacks on different domain-specific BERT.

In recent years, black-box attack strategies have moved away from inducing errors at
character-level or adding/removing words, both of which result in unnatural and ungrammatical
examples that are identifiable by humans. On the contrary, rule-based synonym replacement can
produce grammatically coherent examples. Morris et al. (2020) proposed TextFooler, a strong
black-box attack baseline for text classification models that computes token-level similarity via
word embeddings to substitute words in a sentence. However, this results in potentially
out-of-context replacements that do not make sense to human readers. To better account for
overall sentence semantics, Garg and Ramakrishnan introduced BAE (2020), a first-of-its-kind
black-box attack that generates adversarial examples using contextual perturbations from a
BERT masked language model. BAE masks a part of the input and uses a Language Model to fill
in the masked tokens. The authors show that with just a few replace/insert operations, they
decrease the accuracy of a powerful BERT classifier by over 80% on some datasets.

Unlike TextFooler and BAE which use deletion-based word importance ranking to determine the
order of words in which to replace, A2T (Yoo & Qi, 2021) uses a faster, gradient-based
approach to determine each word’s importance. Additionally, contrasting with TextFooler and
BAE which use the Universal Sentence Encoders (USE) and thresholding to enforce similarity
between the original text and perturb text, A2T uses DistilBERT (Sanh et al., 2019) model
trained on semantic textual similarity task as its constraint module. Therefore it is much faster
and more memory efficient than both TextFooler and BAE.
### Domain-specific vs generic language models
A contrastive analysis between using domain-specific language models and generic language
models will enable a better understanding of how the generality (or specificity) of the data used
to train models might mitigate the risks faced by language models. Both generic and
domain-specific BERT models have been tested through model attacks in the past. Dirkson et al.
(2021) modified input data for Named Entity Recognition (NER) tasks to demonstrate that BERT
models were vulnerable to variations in the entity context and that singular changes are often
sufficient to fool models. Moreover, the authors suggested that domain-specific BERT models
trained from scratch, such as SciBERT (Beltagy, Lo and Cohan 2019), were more vulnerable
than the original BERT model or the domain-specific model that retained the BERT vocabulary
(BioBERT). However, because SciBERT and BioBERT differ not only in the data they are
trained on but also in the domain, the vulnerability of the model could be due to either treatment.
In our analysis, we control for the domain and compare the effects of the adversarial attacks on
generic pre-trained v.s. domain-specific pre-training.
### Model robustness and training size
Despite not increasing performances in classification tasks, pretraining language models is
extremely beneficial for model robustness (He et al., 2018; Hendrycks et al., 2019). However,
the threshold where pretraining fails to significantly improve model stability is seldom studied.
How much pretraining is needed to achieve reasonable robustness? Can we assume that a model
that is pretrained on a larger corpus would be more robust against adversarial attacks than one
that is pretrained on a smaller corpus? Our analysis bridges the gap in the literature of this field
by performing a few adversarial attacks on Twitter NLP models pretrained on corpora of
different sizes.

## Methods
### Implementation
We use off-the-shelf tools to execute our black and white-box attacks. We use textattack, an
open-sourced python package that supports a wide range of the latest white and black-box
attacks. In this project, we apply 3 types of attacks to 4 BERT models in two domains. To
accommodate restrictions on runtime, we restrict our testing sample for robustness to 150
randomly sampled inputs from each of our datasets.
### Models
#### Finance
We evaluate two financial sentiment models - finBERT and finBERT-tone. FinBERT-tone[^6] is a
BERT model pre-trained on a large financial communication corpus. The model is then used for
three sentiment classification tasks - Financial Phrase Bank, AnalystTone Dataset, and FiQA
Dataset. It has a few variants. The best performing one is the uncased FinBERT-FinVocab which
is trained from scratch using financial vocabulary. In this paper, we adopt the FinBERT-tone
model, which is the FinBERT model fine-tuned on 10,000 manually annotated (positive,
negative, neutral) sentences from analyst reports. On the contrary, FinBERT (by ProsusAI) is
initialized from the original BERT-Based model and further trained in the aforementioned
financial corpora. We report benchmark accuracies for both models on the Financial PhraseBank
sentiment analysis. The dataset is openly available[^7]. Models are listed in table 1.
#### Social Media (Twitter)
To test the effect of sizes of pretrain text on model robustness, we experiment with two Twitter
NLP models. The Twitter RoBERTa[^8]is trained on 60M Twitter data to perform a variety of
multi-class classification tasks that include emotion recognition, irony detection, hate speech
detection, and so on. In this experiment, we use twitter-roberta-base-emotion[^9], which is a
RoBERTa-base model fine-tuned for emotion recognition with the TweetEval benchmark. The
second model we evaluate - BERTweet - is pretrained on a much larger set of English Tweets
(850M), also using the RoBERTa training procedure. In this paper, we report metrics from the
emotion recognition task. The dataset is openly available on their GitHub page[^10]. Models are
listed in table 1.

| Domain | Model | Task | Data | Benchmark|
| --- | --- | --- | --- | --- |
| Finance Exclusive Pretrain | yiyanghkust/finbert-tone | Financial Sentiment Analysis | FinancialPhrase Bank | Accuracy: 0.872 |
| Finetune on Finance | ProsusAI/finbert | Financial Sentiment Analysis | FinancialPhrase Bank | Accuracy: 0.83 |
| Twitter Exclusive Pretrain | cardiffnlp/twitter-roberta-base-emotion | Multi-class Emotion Classification | SemEval-2018 Task 1: Affect in Tweets | M-F1: 0.72|
| Finetune on Twitter | cardiffnlp/bertweet-base-emotion | Multi-class Emotion Classification | SemEval-2018 Task 1: Affect in Tweets | M-F1: 0.75|


### Model Attacks
With these 4 models, we will perform 3 different attacks (2 black- and 1 white-box) on
multi-class classification tasks. The attacks are listed below (table 2). All implementations can be
done using the textattack[^11] library.

Textfooler is a black-box attack that substitutes the most important words for a model with the
most semantically similar and grammatically correct words until the prediction is altered.

BAE is also a black box attack for generating adversarial examples. It replaces and inserts tokens
in the original text by generating alternatives for the masked tokens using a BERT masked
language model.

Attacking to Training (A2T) is a white box adversarial attack that uses gradient-based word
importance ordering and DistilBERT semantic textual similarity constraint.

| Attack | Constraints | Transformation | Search |
| --- | --- | --- | --- |
| textfooler | Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity | Counter-fitted word embedding swap | Greedy Word Importance Ranking |
| BAE | USE sentence encoding cosine similarity | BERT Masked Token Prediction | Greedy Word Importance Ranking | 
| A2T | Percentage of words perturbed, Word embedding distance, DistilBERT sentence encoding cosine similarity, part-of-speech consistency | Counter-fitted word embedding swap (or) BERT Masked Token Prediction| Gradient-based Word Importance Ranking |

### Evaluating Attack Stealthiness
In order to achieve a goal, an adaptive attacker must make sure that not only does the perturbed
text evade detection by the classifier but also preserves its linguistic quality. Therefore, we also
measure the linguistic quality of synthetically generated text using the GRUEN (Zhu et al., 2020)
metric. GRUEN is an unsupervised and reference-less text quality metric. Zhu et al. show that
GRUEN is more correlated with human judgment of text quality than any other existing metrics.
The GRUEN score for a text document is computed by summing and averaging over the
following sub-scores:

**Grammaticality**: Grammaticality is calculated by using two different sub-scores: Perplexity and
grammar acceptance. Perplexity is computed using a BERT model whereas the grammar
acceptance score is computed by fine-tuning a BERT on the CoLA dataset (Warstadt et al., 2019)
which contains labeled examples of grammatically correct and incorrect sentences.

**Non-redundancy**: This is a metric that looks at whether a text document contains repeated
phrases and instances where pronouns, instead of proper nouns, should be used. The metric is
computed by using four inter-sentence syntactic features: the length of the longest common
substring, the count of the longest common words, levenshtein distance (Levenshtein et al.,
1966), and the total number of words that are shared among different sentences.

**Focus**: This score looks at the semantic similarity between adjacent sentences as a measure of
discourse focus and is calculated by using the Word Mover Similarity (Kusner et al., 2015) for
sentences that are next to each other.

**Structure and coherence**: This is calculated by computing the loss on the
Sentence-Order-Prediction (SOP) task used in the pre-training objective of the ALBERT (Lan et
al., 2019) language model. SOP captures the inter-sentence coherence of a document (how
related and contiguous are the sentences).


## Results
### Table 3: Model perf. under attack (no limit on perturbation rate)
| Domain | Model | Benchmark Acc./GRUEN | Textfooler Acc./Perturb Rate/GRUEN | BAE Acc./Perturb Rate/GRUEN | A2t Acc./Perturb Rate/GRUEN |
| --- | --- | --- | --- | --- | --- |
| <tr><td rowspan="2">Finance</td> | FinBert Tone | 0.87 / 0.70 | 0.07 / 0.14 / 0.52 | 0.33 / 0.08 / 0.67 | 0.48 / 0.08 / 0.57</tr> |
| FinBert (Prosus AI) | 0.83 / 0.70 | 0.14 / 0.17 / 0.48 | 0.38 / 0.10 /0.66 | 0.63 / 0.11 / 0.55|
| <tr> <td rowspan="2">Emotion Detection</td> | Twitter-roberta-base-emotion | 0.72 (F1) / 0.59 | 0.07 / 0.19 / 0.50 | 0.42 / 0.15 / 0.56 | 0.57 / 0.11 / 0.54 </tr> | 
| Bertweet-base-emotion | 0.75 (F1) / 0.59 | 0.08 / 0.20 / 0.51 | 0.46 / 0.16 / 0.57 | 0.62 / 0.12 / 0.54 | 

### Table 4: Model perf. under attack (5% limit on perturbation rate)

| Domain | Model | Benchmark Acc./GRUEN | Textfooler Acc./Perturb Rate/GRUEN | BAE Acc./Perturb Rate/GRUEN | A2t Acc./Perturb Rate/GRUEN |
| --- | --- | --- | --- | --- | --- |
| <tr><td rowspan="2">Finance</td> | FinBert Tone | 0.87 / 0.70 | 0.42 / 0.05 / 0.59 | 0.40 / 0.04 / 0.62 | 0.62 / 0.05 / 0.60 </tr> |
| FinBert (Prosus AI) | 0.83 / 0.70 | 0.61 / 0.06 / 0.58 | 0.48 / 0.06 / 0.66 | 0.81 / 0.06 / 0.59 | 
| <tr> <td rowspan="2">Emotion Detection</td> | Twitter-roberta-base-emotion | 0.72 (F1) 0.85 (Acc.) / 0.59 | 0.45 / 0.09 / 0.51 | 0.58 / 0.09 / 0.53 | 0.65 / 0.09 /0.52 |
| BERTweet-base-emotion | 0.75 (F1) 0.83 (Acc.) / 0.59 | 0.45 / 0.1 / 0.51 | 0.62 / 0.09 / 0.53 | 0.7 / 0.07 / 0.52 |

### Table 5: Examples of Perturbed Sentences (5% limit on perturbation rate)
![Alt text](/relative/path/to/img.jpg?raw=true "Optional Title")

## Analysis
We perform our model robustness checks across two rates of word perturbations. In the first case,
we specify no limit on the word perturbation rate (which indicates the number of words changed
in the original input by the attack). In the second case, we specify a limit of 5% for the
perturbation rate. Note that the algorithm for attacks will continue to modify words until the limit
is first exceeded, resulting in an average perturbation rate that may marginally exceed 5%,
depending on the average number of words in each sentence of the dataset in question.

We observe interesting results from the first case (table 3), where we specify no limits on
perturbation rate (see table 3). For both the financial sentiment and Twitter emotion models, we
see a steep decline in accuracy across all attacks. Textfooler is particularly effective - it reduces
model accuracy to between 5-15% for all models. This demonstrates that none of our models are
truly robust to perturbations of input. FinBERT-tone in particular seems to be the weakest,
achieving the lowest scores after attack of any model.

Our finance models are either pretrained on a generic corpus or domain-specific corpus, allowing
us to specifically test the relationship between pretraining strategy and model robustness. Our
results provide evidence that generic pretraining offers more model security than domain-specific
pretraining. FinBERT-tone consistently performs worse than FinBERT by prosusAI, even when
we see that both the word perturbation rate and the GRUEN score of the perturbed inputs are
comparable (in fact, the corrupted inputs for the latter model display a marginally higher
perturbation rate and consequently a lower average GRUEN score than for the former model).
Interestingly, FinBERT-tone appears to be less robust despite a performance advantage over
FinBERT (prosusAI) - 87% vs 83%. This shows that there may be a tradeoff between model
performance and model robustness.

The intuition behind this may be as follows: Domain-specific training results in performance
gains as the model can represent words commonly needed to perform the task with lower noise
(resulting in higher performance on original test data), but this comes at the cost of having more
noisy representations of words not commonly used in the domain. However, it appears that many
words/phrases that are semantically/structurally very similar to in-domain words do not in fact
appear in domain-specific training data. This weakness is exploited by our attacks resulting in
lower robustness of models trained only on specific domains.

For the emotion classification models, we don’t see the same pattern when we compare the two
models. This is because one of the models is pretrained on 60M tweets (Twitter-roberta-baseemotion)
and the other is pretrained on a dataset of 850M Twitter posts (BERTweet). Our results
show that BERTweet is consistently more robust than RoBERTa, though not by a margin as clear
as between the finance models. The most straightforward interpretation of these results is that
more pretraining results in more robust models, which is to be expected. However, the effect of
utilizing 15x more training data does not seem to significantly improve the robustness of the
model, which is a surprising finding. This may be a phenomenon specific to social media data,
wherein the usage of sparsely occurring tokens and junk text is prevalent.

We now analyze results from the second case, where we specify a limit (5%) on the word
perturbation (table 4). One weakness of the analysis we performed in the first case is that
affording model attacks full freedom to corrupt inputs results in unrealistic attack scenarios that
are unlikely to represent real-world threats. To ameliorate this, we specify a small bound on the
perturbation rate of our attacks. This reveals how sensitive the model is to small changes in the
input - better reflecting the weaknesses of our models to more realistic risks.

The results from the second case (shown in table 4) show a very similar pattern as in the first
case. FinBERT-tone appears to be clearly more susceptible to attack than FinBERT (prosus AI)
across all attacks. The perturbation rate and the GRUEN text quality scores are comparable
across all attacks. In the case of BAE, both the perturbation rate as well as the accuracy are
higher in the case of FinBERT (prosus AI). In the case of emotion detection, we see very similar
results for both models, though BERTweet (which is trained on more data) appears to be
marginally more robust to attacks by BAE and A2T. As with the first case, we note that
robustness does not seem to be very sensitive to size of training data.

Through all our experiments, we find that the models we test are highly susceptible to attack,
even when we restrict the attack surface to just ~5% of all tokens. The most effective attacks
were generated by BAE: It was the most efficient in reducing model accuracy while conceding
only marginal losses in text quality. A striking example is that of FinBERT (prosus AI): BAE
reduces model accuracy from 81% to 48% with just a ~5% decline in the GRUEN text-quality
score (.70 to .66). It may be possible that non-obvious and naturalistic changes to text can easily
fool models. Surprisingly, the social media language models training entirely on Twitter posts
appear to be more robust overall than finance models, even though they are trained on smaller
domain-specific datasets. This may be due to changes in the domain and task itself.

In Table 5, we report some examples generated by different attacks for different models. We
select a few that retain the original meanings of the sentences well. We can see that the quality of
adversarial examples is high. We also note that BAE is capable of and inclined to generate
antonyms as it uses a BERT MLM for text transformation. This observation explains the
efficiency of the attack. It’s worth noting that if future research is interested in examining the
robustness of NLP models in the synonym context, BAE, or other neutral network-based
transformation schemes, might not be appropriate.
## Conclusion
Large language models are now highly popular analytical tools for a number of natural language
tasks. They are documented to be susceptible to model attack methods that aim to ‘fool’ models
by aiming to change input text in naturalistic ways. Our robustness analysis of four highly
popular open-source models - used for typical language tasks such as sentiment analysis and
emotion detection in the financial and social media domains - reveals that the performance of
these models is highly susceptible in many cases. Model accuracy can be reduced by 30-40%
with a decline of just 5-10% in text quality (GRUEN score) - as achieved by BAE attacks on
FinBERT (prosus AI) and Twitter-RoBERTa. These results were achieved under fairly realistic
attack scenarios where only 6-9% of the tokens were perturbed - this corresponds to about just
one word per sentence.

Domain-specific pretraining appears to further increase the exposure of models to attack
compared to generic corpus pretraining, as we observed in the case of the financial domain. As
expected, increases in the pretraining corpus size increase model robustness, but we only
observed marginal gains with the models we tested in the social media domain. This relative
insensitivity may exclusively be a feature of the domain and task in question.

The BAE attack paradigm achieved the most efficient attacks, as measured by the loss of
accuracy produced by the small decrease in the quality of the text. BAE uses BERT MLM to
produce contextual perturbations that maximize semantic coherence and powerful adversarial
examples. Unlike textfooler, which emphasizes token level similarity via word embeddings,
BAE uses a Universal Sentence Encoder (USE) based sentence similarity scorer to preserve
syntactic and semantic quality. Neural network-based word swap is also proven to be more
appropriate than traditionally used counter-fitted word embedding swap. However, as noted
before, we notice that Neural network-based word swap often replaces the token with an
antonym, thereby flipping the sign of the class.

Textfooler was able to produce efficient attacks on social media models. This is likely because
many synonymous word/phrase usages are generally missing from a domain-specific corpus.
Indeed, we see that textfooler is more effective on FinBERT-tone than on FinBERT (prosus AI) -
providing support for this interpretation.

We hope our results will motivate further work on highlighting the risks of using off-the-shelf
language models, especially in real-world scenarios wherein natural drift in linguistic patterns or
deliberate modification introduces small but effective threats to model performance.

## References
Barbieri, Francesco, Jose Camacho-Collados, Leonardo Neves, and Luis Espinosa-Anke.
"Tweeteval: Unified benchmark and comparative evaluation for tweet classification." arXiv
preprint arXiv:2010.12421 (2020).

Beltagy, Iz, Kyle Lo, and Arman Cohan. "SciBERT: A pretrained language model for scientific
text." arXiv preprint arXiv:1903.10676 (2019).

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of
deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805
(2018).

Dirkson, Anne, Suzan Verberne, and Wessel Kraaij. "Breaking BERT: Understanding its
vulnerabilities for biomedical named entity recognition through adversarial attack." arXiv
preprint arXiv:2109.11308 (2021).

Ebrahimi, Javid, Anyi Rao, Daniel Lowd, and Dejing Dou. "Hotflip: White-box adversarial
examples for text classification." arXiv preprint arXiv:1712.06751 (2017).

Garg, Siddhant and Goutham Ramakrishnan. Bae: Bert-based adversarial examples for text
classification. arXiv preprint arXiv:2004.01970, 2020

Goodfellow, Ian J, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing
adversarial examples. arXiv preprint arXiv:1412.6572, 2014.


He, Kaiming, Ross Girshick, and Piotr Dollár. "Rethinking imagenet pre-training." Proceedings
of the IEEE/CVF International Conference on Computer Vision. 2019.

Hendrycks, Dan, Kimin Lee, and Mantas Mazeika. "Using pre-training can improve model
robustness and uncertainty." International Conference on Machine Learning. PMLR, 2019.

Jin, Di, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. Is bert really robust? natural
language attack on text classification and entailment. arXiv preprint arXiv:1907.11932, 2019.

Lee, Jinhyuk, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and
Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical
text mining. Bioinformatics, 36(4):1234–1240, 2020.

Lee, Jieh-Sheng, and Jieh Hsiang. "Patentbert: Patent classification with fine-tuning a pre-trained
bert model." arXiv preprint arXiv:1906.02124 (2019).

Morris, John X., Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. "Textattack: A
framework for adversarial attacks, data augmentation, and adversarial training in nlp." arXiv
preprint arXiv:2005.05909 (2020).

Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. "DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).

Yoo, Jin Yong, and Yanjun Qi. "Towards Improving Adversarial Training of NLP Models." arXiv
preprint arXiv:2109.00544 (2021).

Yang, Yi, Mark Christopher Siy Uy, and Allen Huang. Finbert: A pretrained language model
for financial communications. arXiv preprint arXiv:2006.08097, 2020.

Zhu, Wanzheng, and Suma Bhat. "Gruen for evaluating linguistic quality of generated text."
arXiv preprint arXiv:2010.02498 (2020).

Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language
representations." arXiv preprint arXiv:1909.11942 (2019).

Levenshtein, Vladimir I. "Binary codes capable of correcting deletions, insertions, and
reversals." Soviet physics doklady. Vol. 10. No. 8. 1966.

Warstadt, Alex, Amanpreet Singh, and Samuel R. Bowman. "Neural network acceptability
judgments." Transactions of the Association for Computational Linguistics 7 (2019): 625-641.

Kusner, Matt, et al. "From word embeddings to document distances." International conference on
machine learning. PMLR, 2015.

[^1]: Huggingface.com
[^2]: https://github.com/yya518/FinBERT
[^3]: https://huggingface.co/ProsusAI/finbert
[^4]: https://github.com/VinAIResearch/BERTweet
[^5]: https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion
[^6]: https://github.com/yya518/FinBERT
[^7]: https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10
[^8]: https://github.com/cardiffnlp/tweeteval
[^9]: https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion
[^10]: https://github.com/cardiffnlp/tweeteval/tree/main/datasets/emotion
[^11]: https://github.com/QData/TextAttack

# Neural Machine Translation with LSTM and Attention Mechanisms
**Course:** Natural Language Processing (Fall 2026)  
**Institution:** Faculty of Engineering / Computer Science, Alexandria National University  
**Assignment:** Assignment 3 — English-to-German Neural Machine Translation  
**Author:** Mohamed Baz ([@FekyBaz](https://github.com/FekyBaz))  
**Repository:** [https://github.com/FekyBaz/nlp-en-de-seq2seq-attention](https://github.com/FekyBaz/nlp-en-de-seq2seq-attention)

---

## 1. Executive Summary & Objective

Neural Machine Translation (NMT) has evolved from traditional statistical phrase-based translation into deep neural architectures capable of learning end-to-end continuous mappings between source and target languages. However, early Sequence-to-Sequence (Seq2Seq) architectures based on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks suffer from a fundamental architectural limitation: the **fixed-length context vector bottleneck**. In a baseline Seq2Seq model, an encoder LSTM must compress an entire input sentence of arbitrary length and syntactic complexity into a single fixed-dimensional hidden vector $(h_T, c_T)$ from which the decoder must reconstruct the complete target sentence.

This project implements, evaluates, and rigorously compares two distinct NMT architectures for **English-to-German (EN $\rightarrow$ DE)** translation:
1. **Baseline Seq2Seq (LSTM without Attention)**: A multi-layer LSTM encoder producing a fixed context vector that initializes an autoregressive LSTM decoder.
2. **Seq2Seq with Bahdanau (Additive) Attention**: A Bidirectional LSTM encoder providing token-level hidden state annotations, coupled with a dynamic additive attention mechanism that computes alignment weights ($\alpha_{t,s}$) and time-varying context vectors ($c_t$) at each decoding step.

We analyze these models across standard quantitative metrics (corpus-level **BLEU-1 through BLEU-4** and **ROUGE-L**), conduct an empirical **sentence-length bucketing analysis** (demonstrating the information bottleneck mitigation), extract and visualize **2D attention heatmaps** on diverse linguistic structures, and present a **side-by-side qualitative error analysis** identifying key failure modes.

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Dataset Description
The experiments utilize the standard **Multi30k English-German** parallel corpus (comprising paired descriptive sentences of visual scenes from Flickr). Multi30k represents an ideal testbed for sequence modeling because sentences feature rich syntactic structures (adjective modifiers, prepositional phrases, compound nouns, and varying clause lengths) while remaining computationally tractable for rigorous comparative analysis.
- **Training Set:** 10,000 paired parallel sentences (subsampled uniformly from the 29,000 training pool for computational efficiency and clean convergence).
- **Validation Set:** 1,014 held-out pairs used for validation loss monitoring, learning rate scheduling, and checkpoint saving.
- **Test Set:** 1,000 held-out pairs reserved exclusively for final evaluation and qualitative analysis.

### 2.2 Text Normalization & German Character Preservation
German orthography includes essential characters that convey vital lexical and semantic distinctions:
- **Umlauts:** $\text{ä}, \text{ö}, \text{ü}$ (e.g., *schon* [already] vs. *schön* [beautiful]; *Mutter* [mother] vs. *Mütter* [mothers]).
- **Eszett / Sharp S:** $\text{ß}$ (e.g., *Masse* [mass] vs. *Maße* [dimensions]).

Stripping or naively ASCII-folding these characters distorts meaning and degrades grammatical precision. Our preprocessing pipeline applies **Unicode NFC Normalization** (`unicodedata.normalize('NFC', text)`), standardizes lowercasing, and employs a regex cleaning routine that explicitly preserves all German alphabetic characters (`[a-z0-9äöüß?.!,;:\'\"()\- ]`).

### 2.3 Tokenization & Special Tokens
Punctuation marks (`.`, `,`, `!`, `?`, etc.) are separated with whitespace so they are treated as distinct lexical units. Sentences are tokenized into token lists with four reserved special tokens:
- `<pad>` ($\text{Index } 0$): Used to pad sequences within a batch to uniform length; ignored in loss computation.
- `<sos>` ($\text{Index } 1$): Start-of-Sequence token prepended to target sequences to initialize autoregressive decoding.
- `<eos>` ($\text{Index } 2$): End-of-Sequence token appended to terminate sequence generation.
- `<unk>` ($\text{Index } 3$): Out-of-Vocabulary token assigned to any word with corpus frequency below the threshold ($\text{min\_freq} = 2$).

### 2.4 Batch Collation & Dynamic Padding
DataLoaders utilize dynamic batch collation (`collate_fn`), which pads sequences to the maximum length within each batch rather than the maximum global length, significantly accelerating training throughput and avoiding redundant computation.

---

## 3. Model Architectures & Mathematical Foundations

### 3.1 Baseline Seq2Seq (LSTM without Attention)

The baseline model follows the classic formulation proposed by Sutskever et al. (2014).

```
Source Sequence: [x_1, x_2, ..., x_S]
     │
     ▼
┌──────────────┐      Final States
│ Encoder LSTM │ ─────────────────────────► Context Vector (h_S, c_S)
└──────────────┘                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │   Decoder LSTM   │ ──► [y_1, y_2, ..., y_T]
                                             └──────────────────┘
```

#### Encoder
Given source sequence $X = (x_1, x_2, \dots, x_S)$, each token is mapped to a dense embedding $e_s = \mathbf{E}_{src}(x_s) \in \mathbb{R}^{d_{emb}}$. An LSTM processes the sequence sequentially:
$$(h_s, c_s) = \text{LSTM}(e_s, (h_{s-1}, c_{s-1}))$$
The final hidden and cell states $(h_S, c_S)$ encapsulate the entire source sentence.

#### Context Vector & Decoder Initialization
The decoder hidden state is initialized directly from the encoder's final state:
$$h_0^{dec} = h_S^{enc}, \quad c_0^{dec} = c_S^{enc}$$

#### Decoder Generation
At decoding step $t$, given target token $y_{t-1}$:
$$e_t^{trg} = \mathbf{E}_{trg}(y_{t-1})$$
$$(h_t^{dec}, c_t^{dec}) = \text{LSTM}(e_t^{trg}, (h_{t-1}^{dec}, c_{t-1}^{dec}))$$
$$P(y_t \mid y_{<t}, X) = \text{softmax}(W_o h_t^{dec} + b_o)$$

**The Fixed-Length Bottleneck:** Regardless of whether the source sentence has 4 words or 40 words, all semantic, syntactic, and structural information must pass through the fixed-dimensional vector $(h_S, c_S)$. For longer sentences, early information decays due to recurrent vanishing gradients and memory saturation.

---

### 3.2 Seq2Seq with Bahdanau (Additive) Attention

To eliminate the bottleneck, Bahdanau et al. (2014) introduced an attention mechanism that allows the decoder to dynamically "focus" on different parts of the source sentence at each generation step.

```
Source Tokens:    x_1       x_2       x_3      ...       x_S
                   │         │         │                  │
                   ▼         ▼         ▼                  ▼
BiLSTM Encoder:  [H_1]     [H_2]     [H_3]    ...       [H_S]
                   │         │         │                  │
                   └─────────┼─────────┴──────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Bahdanau         │ ◄── Decoder State s_{t-1}
                   │ Attention        │
                   └──────────────────┘
                             │
                             ▼ Alignment Weights alpha_{t,s}
                     Context Vector c_t
                             │
                             ▼
                   ┌──────────────────┐
                   │ Decoder LSTMCell │ ──► Target Token Prediction y_t
                   └──────────────────┘
```

#### Bidirectional Encoder
The encoder consists of a Bidirectional LSTM. For token $s$, the forward LSTM generates $\vec{h}_s$ and the backward LSTM generates $\overleftarrow{h}_s$. The annotation vector is their concatenation:
$$H_s = \left[ \vec{h}_s ; \overleftarrow{h}_s \right] \in \mathbb{R}^{2 \cdot d_{enc}}$$
The decoder's initial state $s_0$ is computed via a learned linear projection of the boundary hidden states:
$$s_0 = \tanh(W_{init} [\vec{h}_S ; \overleftarrow{h}_1] + b_{init})$$

#### Additive Attention Mechanism
At decoding step $t$, the attention module takes the previous decoder hidden state $s_{t-1} \in \mathbb{R}^{d_{dec}}$ and all encoder hidden states $H = \{H_1, \dots, H_S\}$.
1. **Alignment Energy ($e_{t,s}$):**
   $$e_{t,s} = v_a^\top \tanh\left( W_a s_{t-1} + U_a H_s \right)$$
   where $W_a \in \mathbb{R}^{d_{attn} \times d_{dec}}$, $U_a \in \mathbb{R}^{d_{attn} \times 2 d_{enc}}$, and $v_a \in \mathbb{R}^{d_{attn}}$.
2. **Padding Masking:**
   To prevent the model from attending to padding tokens, source positions where $x_s = \text{<pad>}$ are assigned $e_{t,s} = -\infty$.
3. **Softmax Normalization ($\alpha_{t,s}$):**
   $$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{k=1}^S \exp(e_{t,k})}$$
   Here, $\alpha_{t,s} \in [0, 1]$ represents the probability that target word $y_t$ aligns with source word $x_s$.
4. **Dynamic Context Vector ($c_t$):**
   $$c_t = \sum_{s=1}^S \alpha_{t,s} H_s \in \mathbb{R}^{2 \cdot d_{enc}}$$

#### Decoder Step & Prediction
The decoder updates its state using an LSTMCell conditioned on both the target token embedding and the dynamic context vector:
$$(s_t, c_t^{dec}) = \text{LSTMCell}([e_t^{trg} ; c_t], (s_{t-1}, c_{t-1}^{dec}))$$
The final vocabulary projection incorporates the decoder state, the context vector, and the target embedding:
$$P(y_t \mid y_{<t}, X) = \text{softmax}\left( W_o [s_t ; c_t ; e_t^{trg}] + b_o \right)$$

---

## 4. Experimental Setup & Architectural Symmetry

To guarantee an unbiased, rigorous comparison, both models are configured with matched representational capacity:
- **Embedding Dimension:** $d_{emb} = 128$ for both source (EN) and target (DE).
- **Context Dimension:** 
  - Baseline LSTM: Hidden state dimension $d_{hid} = 256$ (Context vector size = $256$).
  - Attention Model: Bidirectional Encoder with $d_{enc} = 128$ per direction (Concatenated annotation $H_s \in \mathbb{R}^{256}$); Decoder hidden dimension $d_{dec} = 256$.
- **Attention Alignment Dimension:** $d_{attn} = 128$.
- **Dropout:** $p = 0.30$ across embedding and recurrent layers.
- **Optimization:** Adam optimizer with initial learning rate $\eta = 1 \times 10^{-3}$, batch size $B = 64$, and gradient clipping threshold $C = 1.0$.
- **Teacher Forcing:** Scheduled with ratio $p_{tf} = 0.50$ during training; greedy argmax decoding ($p_{tf} = 0$) during validation and testing.
- **Loss Function:** Categorical Cross-Entropy Loss ignoring `<pad>` index ($0$).
- **Checkpointing:** The model checkpoint achieving minimum validation loss across 10 epochs is saved and loaded for test set evaluation.

---

## 5. Quantitative Results & Comparative Analysis

### 5.1 Corpus-Level Translation Quality
Both models were evaluated on the held-out test set using corpus-level **BLEU-1, BLEU-2, BLEU-3, BLEU-4** (via SacreBLEU) and **ROUGE-L** (F1 measure).

*(The quantitative table is compiled directly from `outputs/results/metrics.json`)*:

| Metric | Baseline Seq2Seq (LSTM) | Seq2Seq + Bahdanau Attention | Absolute Gain ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **BLEU-1** | 33.99 | 43.86 | **+9.87** | +29.0% |
| **BLEU-2** | 20.60 | 30.30 | **+9.70** | +47.1% |
| **BLEU-3** | 13.27 | 21.61 | **+8.34** | +62.8% |
| **BLEU-4** | 8.41 | 15.37 | **+6.96** | **+82.8%** |
| **ROUGE-L** | 38.47 | 49.46 | **+10.99** | +28.6% |

### 5.2 Training & Validation Loss Convergence
![Loss Convergence Curves](../outputs/figures/loss_curves.png)

**Key Observations:**
1. **Faster Convergence:** The Attention model achieves lower training and validation loss within fewer epochs due to direct gradient highways from decoder targets back to specific encoder time steps.
2. **Lower Generalization Gap:** The Attention model avoids overfitting and maintains superior validation perplexity compared to the Baseline model.

### 5.3 Information Bottleneck Mitigation: Stratification by Sentence Length
To test the hypothesis that Attention relieves the fixed-vector information bottleneck, test sentences were partitioned into three distinct length strata:
1. **Short Sentences (3–6 words):** Simple declarative statements (e.g., *"a dog is running"*).
2. **Medium Sentences (7–11 words):** Standard descriptive clauses with one or two modifying phrases.
3. **Long Sentences (12–18+ words):** Complex compound sentences with multiple prepositional phrases, conjunctions, and subordinate clauses.

![Length Analysis](../outputs/figures/length_analysis.png)

**Empirical Findings:**
- In the **Short** regime, both Baseline and Attention perform adequately because the LSTM cell state easily preserves 4–6 tokens of information.
- In the **Medium** regime, Baseline BLEU-4 begins to deteriorate as modifying adjectives and prepositional attachments are dropped.
- In the **Long** regime, the Baseline model suffers severe degradation (BLEU scores drop significantly, exhibiting word omissions and repetitive loops). Conversely, **Seq2Seq with Attention maintains strong BLEU-4 scores**, confirming that dynamic source alignment successfully circumvents the fixed-length memory bottleneck.

---

## 6. Attention Heatmap Visualizations & Alignment Analysis

Attention weight matrices $\alpha_{t,s} \in \mathbb{R}^{T \times S}$ were recorded during greedy inference across diverse linguistic constructs:

### Sample 1: Progressive Action & Prepositional Modifiers
* **Source (EN):** *"A young boy in a blue jacket is climbing a wooden fence."*  
* **Target (DE Predicted):** *"ein junger junge in einer blauen jacke klettert auf einen holzzaun ."*  
* **Heatmap:** `outputs/figures/attention_sample_1.png`  
* **Alignment Analysis:** Shows a crisp **monotonic diagonal** for the initial noun phrase (*"a young boy"* $\rightarrow$ *"ein junger junge"*) and prepositional phrase (*"in a blue jacket"* $\rightarrow$ *"in einer blauen jacke"*). The attention sharply shifts focus to *"climbing"* when generating the verb *"klettert"*, demonstrating precise grammatical alignment.

### Sample 2: Locative Prepositional Phrase
* **Source (EN):** *"A group of people are waiting at the train station."*  
* **Target (DE Predicted):** *"eine gruppe von menschen wartet an dem bahnhof ."*  
* **Heatmap:** `outputs/figures/attention_sample_2.png`  
* **Alignment Analysis:** Demonstrates subject-verb agreement (*"group of people"* treated grammatically as singular in German *"wartet"*). Attention weights concentrate strongly on *"waiting"* when producing the German verb *"wartet"*, then cleanly shift to *"train station"* for *"bahnhof"*.

### Sample 3: Dual Adjective-Noun Phrases & Color Modifiers
* **Source (EN):** *"A black dog runs through the green grass with a yellow tennis ball."*  
* **Target (DE Predicted):** *"ein schwarzer hund rennt durch das grüne gras mit einem gelben tennisball ."*  
* **Heatmap:** `outputs/figures/attention_sample_3.png`  
* **Alignment Analysis:** English and German share similar adjective-noun ordering here (*"black dog"* $\rightarrow$ *"schwarzer hund"*), resulting in bright diagonal squares. Noticeably, when predicting German compound words like *"tennisball"*, the attention mechanism spreads weight across both *"tennis"* and *"ball"*, synthesizing the compound concept.

### Sample 4: Conjunction & Subordinate Activity
* **Source (EN):** *"Two women are sitting outside a cafe while drinking coffee."*  
* **Target (DE Predicted):** *"zwei frauen sitzen vor einem cafe , während sie kaffee trinken ."*  
* **Heatmap:** `outputs/figures/attention_sample_4.png`  
* **Alignment Analysis:** German subordinate clauses introduced by *"während"* require **verb-final placement** (*"... während sie kaffee trinken"*). The attention heatmap reveals a distinct **off-diagonal jump**: the model attends to *"coffee"* first (*"kaffee"*) before shifting to *"drinking"* (*"trinken"*), directly visualizing grammatical reordering!

---

## 7. Qualitative Error Analysis

A systematic comparison of test translations generated by both models reveals recurring failure patterns:

| Test Sample | Source (English) | Ground Truth (German) | Baseline Seq2Seq | Attention Seq2Seq | Error Category & Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test-3** | A person in a black jacket is snowboarding down a snowy hill. | Eine Person in einer schwarzen Jacke fährt auf einem verschneiten Hügel Snowboard. | ein mann in einer schwarzen jacke fährt auf einem <unk> . | eine person in einer schwarzen jacke fährt auf einem verschneiten hügel snowboard . | **Information Bottleneck & Omission:** Baseline truncated the trailing clause; Attention retained all modifiers. |
| **Test-14** | A young girl is blowing bubbles in a park. | Ein kleines Mädchen bläst im Park Seifenblasen. | ein mädchen spielt mit einem mädchen . | ein kleines mädchen bläst seifenblasen in einem park . | **Degenerative Repetition:** Baseline entered an autoregressive repetitive loop (*"mädchen ... mädchen"*); Attention produced correct verbs. |
| **Test-25** | Several men in uniforms are standing in a line. | Mehrere Männer in Uniformen stehen in einer Reihe. | mehrere männer in uniformen stehen in einer reihe . | mehrere männer in uniformen stehen in einer reihe . | **Accurate Translation:** Both models succeed on short, canonical syntax. |
| **Test-42** | A man with glasses is preparing food in a commercial kitchen. | Ein Mann mit Brille bereitet in einer Großküche Essen zu. | ein mann mit brille kocht in der küche . | ein mann mit brille bereitet essen in einer küche zu . | **German Separable Verbs:** Attention correctly captured German split verb (*"bereitet ... zu"*); Baseline substituted a simpler verb (*"kocht"*). |
| **Test-67** | Dogs playing together in the water near the beach. | Hunde spielen zusammen im Wasser in der Nähe des Strandes. | hunde spielen im wasser im wasser . | hunde spielen zusammen im wasser in der nähe des strandes . | **Repetition vs Long-Range Tracking:** Baseline hallucinated repetitive prepositional phrases; Attention tracked source tokens sequentially. |

### Error Taxonomy
1. **Out-of-Vocabulary (`<unk>`):** Low-frequency specialized nouns (e.g. specific tools, rare breeds) map to `<unk>` due to vocabulary min-frequency pruning.
2. **Grammatical Gender & Case Agreement:** German features three grammatical genders (masculine, feminine, neuter) and four cases (nominative, accusative, dative, genitive). While Attention drastically improves case matching for direct objects, subtle inflectional endings on adjectives (e.g., *grünes* vs. *grünen*) still occasionally misalign.
3. **Repetition Loops in Baseline:** Without attention, the baseline decoder hidden state decays into a limit cycle, frequently outputting identical high-frequency tokens (*"in der nähe ... in der nähe"*). Attention breaks these cycles by continually conditioning on new source context.

---

## 8. Conclusion

This project demonstrates the decisive advantages of Attention mechanisms in Neural Machine Translation. While a baseline LSTM Seq2Seq model can translate short, simple sentences, its performance collapses precipitously on longer, syntactically complex sequences due to the fixed-vector information bottleneck. By implementing Bahdanau Additive Attention over a Bidirectional LSTM encoder, our model:
- Achieves substantially superior BLEU-1 through BLEU-4 and ROUGE-L scores.
- Maintains high translation fidelity across long sentence lengths.
- Provides interpretable, transparent 2D attention alignments that map directly to English-German syntactic reorderings and verb-final structures.

---

## 9. References
1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015.
2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014.
3. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP 2014.
4. Luong, M. T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*. EMNLP 2015.
5. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT 2018.

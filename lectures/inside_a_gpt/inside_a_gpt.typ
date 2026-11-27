#import "@preview/touying:0.7.4": *

#let main_title = [Inside a GPT]
#let main_author = [#link("https://www.bpesquet.fr")[Baptiste Pesquet]]

// 3Blue1Brown lessons used during the session
#let video_attention = "https://www.3blue1brown.com/lessons/attention"
#let video_mlp = "https://www.youtube.com/watch?v=9-Dc0ayAHt8"
#let microgpt_url = "https://karpathy.github.io/2026/02/12/microgpt/"

#import themes.simple: *
#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [#main_title],
  // Uncomment the following line to obtain an animation-free version
  // config-common(handout: true),
)

#set quote(block: true)
#show figure.caption: set text(size: 16pt)
#show table: set align(right)

#title-slide[
  // Institution logos
  #place(
    bottom + center,
    link("https://ensc.bordeaux-inp.fr")[#image("images/ensc_logo.jpg", width: 15%)],
  )
  #title[
    #main_title
  ]

  #v(1em)

  #main_author

  #datetime.today().display("[month repr:long] [year repr:full]")
]

== Outline <touying:hidden>

#show outline: set text(size: 20pt)

#components.adaptive-columns(outline(title: none, depth: 1, indent: 1em))

= From fixed windows to attention

== Where we left off

- Bengio's neural language model: concatenated embeddings of the last $n - 1$ words.
- Limits:
  - fixed window: "The _keys_ that I left on the kitchen table yesterday _are_...";
  - no interactions between words: same embedding for _bank_ next to _river_ and next to _account_.

Idea: let each position compute a _weighted average_ of the previous ones, with weights depending on the words themselves.

== The complete algorithm in 200 lines

- #link(microgpt_url)[microgpt] (Karpathy, 2026): a GPT trained and sampled in pure Python, on top of a #link("https://github.com/karpathy/micrograd")[micrograd]-like `Value` class.
- 4,192 parameters.
- Trained on a dataset of about 32,000 English first names.

👩🏻‍💻 Activity 1: experiment with microgpt.

= Self-attention

== Watch: attention in transformers

#align(horizon + center)[
  🎬 #link(video_attention)[3Blue1Brown — Attention in transformers, step-by-step] (26 min)
]

== Queries, keys and values

- Each token embedding $x_i$ emits three vectors:
  - a _query_ $q_i = x_i W_Q$: what am I looking for?
  - a _key_ $k_i = x_i W_K$: what do I contain?
  - a _value_ $v_i = x_i W_V$: what do I give if chosen?
- With all positions stacked in matrices $Q, K, V$:

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k) + M) V $

- $M$: causal mask, $0$ for past and present positions, $-infinity$ for future ones.

---

=== Why the mask, why the scaling?

- Training processes the $T$ positions of a sequence in parallel ($T$ examples in one pass): without the mask, position $t$ could read the token it has to predict.
- $q dot k$ has variance $d_k$: large scores saturate the softmax. Dividing by $sqrt(d_k)$ brings it back to 1.
- A uniform attention pattern is the causal average of embeddings: a running bag of words.

👩🏻‍💻 Activity 2: implement self-attention with tensors.

== Multi-head attention

- Several heads run in parallel, each with its own $W_Q$, $W_K$, $W_V$ and a smaller dimension $d_k = C \/ h$.
- Different heads can specialize: previous token, beginning of the word, matching quote...
- Head outputs are concatenated, then projected back by $W_O$.
- Cost: the attention pattern has $T^2$ entries per head $==>$ quadratic in the context length.

= The GPT architecture

== Transformer block

- _Attention_: tokens communicate.
- _MLP_ (feed-forward network, $C -> 4C -> C$): each token is processed on its own.
- _Residual connections_: each sub-block only adds a correction to the _residual stream_.
- _Normalization_ (LayerNorm or RMSNorm) before each sub-block.

$ x <- x + "Attention"("Norm"(x)) $
$ x <- x + "MLP"("Norm"(x)) $

== Full model

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: left,
    table.header([*Stage*], [*Operation*], [*Shape*]),
    [Input], [token IDs], [$B times T$],
    [Embeddings], [token embedding + position embedding], [$B times T times C$],
    [Blocks], [$N times$ (attention + MLP, with residuals)], [$B times T times C$],
    [Head], [final norm, then linear unembedding], [$B times T times V$],
    [Output], [softmax over the vocabulary], [next-token distributions],
  ),
)

---

=== Training and evaluation

- Training objective: minimize the cross-entropy of next-token prediction.
- Evaluation: validation perplexity. Nothing new since the previous chapter!

👩🏻‍💻 Activity 3: build and train your own GPT on GPU.

== Counting parameters

Per block, ignoring biases and norms: $4 C^2$ for attention, $8 C^2$ for the MLP.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: left,
    table.header([*Model*], [*Layers*], [*$C$*], [*Parameters*]),
    [microgpt], [1], [16], [4,192],
    [Our GPT (Lab 2)], [6], [192], [≈ 2.7M],
    [GPT-2 small], [12], [768], [124M],
    [GPT-3], [96], [12,288], [175B],
  ),
)

👩🏻‍💻 Activity 4: count parameters, then run ablation and fine-tuning experiments.

= One architecture, three families

== Same block, different choices

Every transformer-based model stacks the same blocks. Families differ by three choices:
- the attention _mask_: causal or bidirectional;
- the training _objective_: next token, or hidden tokens;
- how blocks are _connected_: one stack, or an encoder feeding a decoder.

👩🏻‍💻 Activity 5: remove the mask, then use an encoder and an encoder–decoder.

== Encoders

- No mask: each token sees the whole sentence (_bidirectional_ attention).
- Next-token prediction becomes trivial (copy the next token), so the objective changes: _masked language modeling_, i.e. hide about 15% of the tokens and predict them.
- One contextual vector per token $==>$ pooled into _text embeddings_.
- Uses: classification, named entities, semantic search, retrieval-augmented generation (RAG).
- Examples: BERT (2018), CamemBERT, sentence-transformers.

== Encoder–decoders

- The original Transformer (2017) was designed for translation.
- The _encoder_ reads the source (no mask) once.
- The _decoder_ generates the target autoregressively (causal mask), and looks at the encoder outputs through _cross-attention_: queries from the decoder, keys and values from the encoder.
- Uses: translation, summarization, speech recognition.
- Examples: T5, BART, Marian, Whisper.

== Summary

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: left,
    table.header([], [*Decoder*], [*Encoder*], [*Encoder–decoder*]),
    [Mask], [causal], [none], [none, then causal + cross-attention],
    [Objective], [next token], [masked tokens], [target from source],
    [Uses], [generation, chatbots], [embeddings, classification], [translation, summarization],
    [Examples], [GPT, Llama, Qwen], [BERT, CamemBERT], [T5, Marian, Whisper],
  ),
)

Today, almost all LLMs are decoders. Encoders remain the workhorses for embeddings and classification.

== Key takeaways

- A GPT is embeddings, then $N$ blocks of attention and MLP on a residual stream, then a next-token distribution.
- Attention is a learned, data-dependent weighted average of the past.
- Most parameters live in the MLP blocks and, for large vocabularies, in the embeddings.
- From microgpt to GPT-3: same algorithm, different scale.
- Same block, three families: the mask and the objective decide whether a model generates, understands or transforms text.

🏠 Homework: #link(video_mlp)[3Blue1Brown — How might LLMs store facts] (23 min).

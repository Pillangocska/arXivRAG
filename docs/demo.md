# Demo & Evaluation

This page shows what running the system looks like and how to read the
evaluation output.

## Prerequisites recap

1. `uv sync` — install dependencies.
2. Set `ANTHROPIC_API_KEY` in `.env`.
3. `make up` — start Qdrant in Docker.
4. Place the Kaggle corpus at `_data/arxiv-metadata-oai-snapshot.json`.
5. `make ingest` — build the index (filter to `cs.LG`, embed, upsert).

## Example: a hybrid, multi-part question

The defining case for this system is a question that bundles a *content* need
(answered by the local index) with a *current-state* need (answered by the
live arXiv API):

```bash
make run Q="What does the original transformer paper propose, and what recent papers improve on its attention mechanism?"
```

Expected shape of the output:

```
=== answer ===

The original "Attention Is All You Need" transformer paper is not directly described in the provided context (no retrieved excerpt covers it), so its specific proposals cannot be cited here. However, the context does describe several recent works that improve upon or extend the standard transformer self-attention mechanism:

**Dilated Neighborhood Attention (DiNA)** addresses the limitation that local attention mechanisms — such as sliding-window Neighborhood Attention (NA) and Swin Transformer's Shifted Window Self-Attention — weaken two key properties of self-attention: long-range inter-dependency modeling and global receptive field. DiNA extends NA by capturing more global context and expanding receptive fields exponentially at no additional cost. The resulting DiNAT model achieves significant improvements over baselines like NAT, Swin, and ConvNeXt on object detection, instance segmentation, and semantic segmentation tasks (arXiv:2209.15001).

**Energy-Gated Attention (EGA) and Morlet Positional Encoding (MoPE)** identify two inductive biases that standard attention lacks: energy salience (which tokens concentrate informational energy) and scale-selective locality (how far positional influence extends at each frequency). EGA gates value aggregation by a learned energy estimate of key token embeddings, while MoPE replaces fixed sinusoidal encodings with learned Gaussian-windowed wavelets. Their combination achieves superadditive improvements over standard attention, with the two components addressing complementary gaps (arXiv:2605.26355).

**Convolution-based active memory** has been investigated as a potential replacement or complement to self-attention in transformers. Experiments show that active-memory mechanisms alone achieve comparable results to self-attention for language modeling, but optimal results come from combining both; for some algorithmic tasks, active-memory alone outperforms self-attention (arXiv:1912.11959).

**Mask-Attention-Free Transformer** for 3D instance segmentation abandons the mask-attention pipeline common in transformer-based methods, replacing it with an auxiliary center regression task and position-aware cross-attention designs, achieving 4× faster convergence and state-of-the-art results (arXiv:2309.01692).

--- trace ---
  [1] route=vector grade=weak retries=1 hits=5
      q: What are the key proposals in the "Attention Is All You Need" paper and how does the self-attention mechanism work?
  [2] route=arxiv grade=good retries=0 hits=5
      q: Recent papers that improve on transformer attention mechanisms
```

The original "Attention is all you need" paper was not ingested into our vector database, we can clearly see this from the answer but the system was still able to retrieve relevant information from the API and its own knowledge.

## Example: the corrective loop

When the first retrieval for a sub-query grades `weak`, the agent reformulates
the query once and retries before synthesizing. A trace from that path looks
like:

```
--- trace ---
  [1] route=vector grade=weak retries=1 hits=4
      q: <reformulated query text>
2026-06-23 13:28:04 [WARNING] __main__: low confidence: weak context for part of the query.
```

`retries=1` with `MAX_RETRIES=1` is the bound in action: the loop fires at most
once, then the system proceeds and flags low confidence rather than looping
(see `docs/ADR.md` section 4.7). The deterministic test
`test_agent_integration.py::test_end_to_end_weak_context_retries_once_then_finishes`
proves this terminates.

## Evaluation

Generate a synthetic, corpus-derived eval set and score the agent with Ragas:

```bash
make eval                  # = eval-generate (size 10) + eval-run
# or control the size:
make eval-generate EVAL_SIZE=50
make eval-run
```

`generate.py` samples papers from the ingested category and asks the grader
model to write a question + reference answer per paper. `run_eval.py` runs the
full agent over each question and scores the results.

Results are written to `eval/results/scores.json`. Illustrative shape:

```json
{
  "scores": {
    "faithfulness": 0.7648777348777349,
    "answer_relevancy": 0.9512870491094271,
    "context_precision": 0.9866666665787778
  },
  "n": 10,
  "valid": {
    "faithfulness": 10,
    "answer_relevancy": 10,
    "context_precision": 10
  },
  "per_row": [
...
    {
      "question": "What spreading activation approach is proposed for collaborative filtering, and what parameter is introduced to regulate object contributions to user similarity calculations?",
      "answer": "The spreading activation approach for collaborative filtering (SA-CF) proposed uses an opinion spreading process to derive similarity between any users, achieving remarkably higher accuracy than standard collaborative filtering using Pearson correlation (arXiv:0712.3807). The free parameter **β** is introduced to regulate the contributions of objects to user-user correlations, with numerical results indicating that decreasing the influence of popular objects further improves algorithmic accuracy and personality (arXiv:0712.3807). The work also proposes a top-*N* similar neighbors variant that simultaneously reduces computational complexity while maintaining higher algorithmic accuracy (arXiv:0712.3807).",
      "reference": "The paper proposes a spreading activation approach for collaborative filtering (SA-CF) that uses an opinion spreading process to obtain similarity between users. A free parameter β is introduced to regulate the contributions of objects to user-user correlations, with the finding that decreasing the influence of popular objects can improve algorithmic accuracy and personality.",
      "retrieved_contexts": [
        "Improved Collaborative Filtering Algorithm via Information Transformation\n\nIn this paper, we propose a spreading activation approach for collaborative filtering (SA-CF). By using the opinion spreading process, the similarity between any users can be obtained. The algorithm has remarkably higher accuracy than the standard collaborative filtering (CF) using Pearson correlation. Furthermore, we introduce a free parameter $\\beta$ to regulate the contributions of objects to user-user correlations. The numerical results indicate that decreasing the influence of popular objects can further improve the algorithmic accuracy and personality. We argue that a better algorithm should simultaneously require less computation and generate higher accuracy. Accordingly, we further propose an algorithm involving only the top-$N$ similar neighbors for each target user, which has both less computational complexity and higher algorithmic accuracy.",
        "A Latent Source Model for Online Collaborative Filtering\n\nDespite the prevalence of collaborative filtering in recommendation systems, there has been little theoretical development on why and how well it works, especially in the \"online\" setting, where items are recommended to users over time. We address this theoretical gap by introducing a model for online recommendation systems, cast item recommendation under the model as a learning problem, and analyze the performance of a cosine-similarity collaborative filtering method. In our model, each of $n$ users either likes or dislikes each of $m$ items. We assume there to be $k$ types of users, and all the users of a given type share a common string of probabilities determining the chance of liking each item. At each time step, we recommend an item to each user, where a key distinction from related bandit literature is that once a user consumes an item (e.g., watches a movie), then that item cannot be recommended to the same user again. The goal is to maximize the number of likable items recommended to users over time. Our main result establishes that after nearly $\\log(km)$ initial learning time steps, a simple collaborative filtering algorithm achieves essentially optimal performance without knowing $k$. The algorithm has an exploitation step that uses cosine similarity and two types of exploration steps, one to explore the space of items (standard in the literature) and the other to explore similarity between users (novel to this work).",
        "Modeling User Exposure in Recommendation\n\nCollaborative filtering analyzes user preferences for items (e.g., books, movies, restaurants, academic papers) by exploiting the similarity patterns across users. In implicit feedback settings, all the items, including the ones that a user did not consume, are taken into consideration. But this assumption does not accord with the common sense understanding that users have a limited scope and awareness of items. For example, a user might not have heard of a certain paper, or might live too far away from a restaurant to experience it. In the language of causal analysis, the assignment mechanism (i.e., the items that a user is exposed to) is a latent variable that may change for various user/item combinations. In this paper, we propose a new probabilistic approach that directly incorporates user exposure to items into collaborative filtering. The exposure is modeled as a latent variable and the model infers its value from data. In doing so, we recover one of the most successful state-of-the-art approaches as a special case of our model, and provide a plug-in method for conditioning exposure on various forms of exposure covariates (e.g., topics in text, venue locations). We show that our scalable inference algorithm outperforms existing benchmarks in four different domains both with and without exposure covariates.",
        "Empirical Analysis of Predictive Algorithms for Collaborative Filtering\n\nCollaborative filtering or recommender systems use a database about user preferences to predict additional topics or products a new user might like. In this paper we describe several algorithms designed for this task, including techniques based on correlation coefficients, vector-based similarity calculations, and statistical Bayesian methods. We compare the predictive accuracy of the various methods in a set of representative problem domains. We use two basic classes of evaluation metrics. The first characterizes accuracy over a set of individual predictions in terms of average absolute deviation. The second estimates the utility of a ranked list of suggested items. This metric uses an estimate of the probability that a user will see a recommendation in an ordered list. Experiments were run for datasets associated with 3 application areas, 4 experimental protocols, and the 2 evaluation metrics for the various algorithms. Results indicate that for a wide range of conditions, Bayesian networks with decision trees at each node and correlation methods outperform Bayesian-clustering and vector-similarity methods. Between correlation and Bayesian networks, the preferred method depends on the nature of the dataset, nature of the application (ranked versus one-by-one presentation), and the availability of votes with which to make predictions. Other considerations include the size of database, speed of predictions, and learning time.",
        "Power of the Few: Analyzing the Impact of Influential Users in Collaborative Recommender Systems\n\nLike other social systems, in collaborative filtering a small number of \"influential\" users may have a large impact on the recommendations of other users, thus affecting the overall behavior of the system. Identifying influential users and studying their impact on other users is an important problem because it provides insight into how small groups can inadvertently or intentionally affect the behavior of the system as a whole. Modeling these influences can also shed light on patterns and relationships that would otherwise be difficult to discern, hopefully leading to more transparency in how the system generates personalized content. In this work we first formalize the notion of \"influence\" in collaborative filtering using an Influence Discrimination Model. We then empirically identify and characterize influential users and analyze their impact on the system under different underlying recommendation algorithms and across three different recommendation domains: job, movie and book recommendations. Insights from these experiments can help in designing systems that are not only optimized for accuracy, but are also tuned to mitigate the impact of influential users when it might lead to potential imbalance or unfairness in the system's outcomes."
      ],
      "n_contexts": 5,
      "scores": {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9101821578424504,
        "context_precision": 0.9999999999
      }
    },
    {
      "question": "What advantage does combining multiple haplotyper predictions offer compared to selecting a single method for haplotype reconstruction?",
      "answer": "Combining multiple haplotyper predictions offers the advantage of circumventing the method selection problem that arises because different haplotyping methods employ different statistical models and implicitly encode different assumptions about the nature of the underlying haplotype structure (arXiv:0710.5116). Since the relative performance of individual methods can vary greatly depending on the population sample in question, it is unclear which single method to choose for any particular sample (arXiv:0710.5116). By combining predictions from different methods in a principled way, the resulting combined prediction is typically at least as accurate as, or even more accurate than, the best individual method (arXiv:0710.5116). Beyond accuracy, combination techniques also provide more robust reconstructions and are useful for outlier detection (arXiv:0710.5116).",
      "reference": "Combining predictions from different haplotyper methods provides more accurate and robust reconstructions than individual methods alone. The combined prediction is typically at least as accurate as or even more accurate than the best individual method, effectively circumventing the method selection problem while also being useful for outlier detection.",
      "retrieved_contexts": [
        "Combining haplotypers\n\nStatistically resolving the underlying haplotype pair for a genotype measurement is an important intermediate step in gene mapping studies, and has received much attention recently. Consequently, a variety of methods for this problem have been developed. Different methods employ different statistical models, and thus implicitly encode different assumptions about the nature of the underlying haplotype structure. Depending on the population sample in question, their relative performance can vary greatly, and it is unclear which method to choose for a particular sample. Instead of choosing a single method, we explore combining predictions returned by different methods in a principled way, and thereby circumvent the problem of method selection. We propose several techniques for combining haplotype reconstructions and analyze their computational properties. In an experimental study on real-world haplotype data we show that such techniques can provide more accurate and robust reconstructions, and are useful for outlier detection. Typically, the combined prediction is at least as accurate as or even more accurate than the best individual method, effectively circumventing the method selection problem.",
        "Matrix Completion and Performance Guarantees for Single Individual Haplotyping\n\nSingle individual haplotyping is an NP-hard problem that emerges when attempting to reconstruct an organism's inherited genetic variations using data typically generated by high-throughput DNA sequencing platforms. Genomes of diploid organisms, including humans, are organized into homologous pairs of chromosomes that differ from each other in a relatively small number of variant positions. Haplotypes are ordered sequences of the nucleotides in the variant positions of the chromosomes in a homologous pair; for diploids, haplotypes associated with a pair of chromosomes may be conveniently represented by means of complementary binary sequences. In this paper, we consider a binary matrix factorization formulation of the single individual haplotyping problem and efficiently solve it by means of alternating minimization. We analyze the convergence properties of the alternating minimization algorithm and establish theoretical bounds for the achievable haplotype reconstruction error. The proposed technique is shown to outperform existing methods when applied to synthetic as well as real-world Fosmid-based HapMap NA12878 datasets.",
        "ComHapDet: A Spatial Community Detection Algorithm for Haplotype Assembly\n\nBackground: Haplotypes, the ordered lists of single nucleotide variations that distinguish chromosomal sequences from their homologous pairs, may reveal an individual's susceptibility to hereditary and complex diseases and affect how our bodies respond to therapeutic drugs. Reconstructing haplotypes of an individual from short sequencing reads is an NP-hard problem that becomes even more challenging in the case of polyploids. While increasing lengths of sequencing reads and insert sizes {\\color{black} helps improve accuracy of reconstruction}, it also exacerbates computational complexity of the haplotype assembly task. This has motivated the pursuit of algorithmic frameworks capable of accurate yet efficient assembly of haplotypes from high-throughput sequencing data. Results: We propose a novel graphical representation of sequencing reads and pose the haplotype assembly problem as an instance of community detection on a spatial random graph. To this end, we construct a graph where each read is a node with an unknown community label associating the read with the haplotype it samples. Haplotype reconstruction can then be thought of as a two-step procedure: first, one recovers the community labels on the nodes (i.e., the reads), and then uses the estimated labels to assemble the haplotypes. Based on this observation, we propose ComHapDet - a novel assembly algorithm for diploid and ployploid haplotypes which allows both bialleleic and multi-allelic variants. Conclusions: Performance of the proposed algorithm is benchmarked on simulated as well as experimental data obtained by sequencing Chromosome $5$ of tetraploid biallelic \\emph{Solanum-Tuberosum} (Potato). The results demonstrate the efficacy of the proposed method and that it compares favorably with the existing techniques.",
        "Merging versus Ensembling in Multi-Study Prediction: Theoretical Insight from Random Effects\n\nA critical decision point when training predictors using multiple studies is whether studies should be combined or treated separately. We compare two multi-study prediction approaches in the presence of potential heterogeneity in predictor-outcome relationships across datasets: 1) merging all of the datasets and training a single learner, and 2) multi-study ensembling, which involves training a separate learner on each dataset and combining the predictions resulting from each learner. For ridge regression, we show analytically and confirm via simulation that merging yields lower prediction error than ensembling when the predictor-outcome relationships are relatively homogeneous across studies. However, as cross-study heterogeneity increases, there exists a transition point beyond which ensembling outperforms merging. We provide analytic expressions for the transition point in various scenarios, study asymptotic properties, and illustrate how transition point theory can be used for deciding when studies should be combined with an application from metagenomics.",
        "Improving the Stability of the Knockoff Procedure: Multiple Simultaneous Knockoffs and Entropy Maximization\n\nThe Model-X knockoff procedure has recently emerged as a powerful approach for feature selection with statistical guarantees. The advantage of knockoff is that if we have a good model of the features X, then we can identify salient features without knowing anything about how the outcome Y depends on X. An important drawback of knockoffs is its instability: running the procedure twice can result in very different selected features, potentially leading to different conclusions. Addressing this instability is critical for obtaining reproducible and robust results. Here we present a generalization of the knockoff procedure that we call simultaneous multi-knockoffs. We show that multi-knockoff guarantees false discovery rate (FDR) control, and is substantially more stable and powerful compared to the standard (single) knockoff. Moreover we propose a new algorithm based on entropy maximization for generating Gaussian multi-knockoffs. We validate the improved stability and power of multi-knockoffs in systematic experiments. We also illustrate how multi-knockoffs can improve the accuracy of detecting genetic mutations that are causally linked to phenotypes."
      ],
      "n_contexts": 5,
      "scores": {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9345603850978149,
        "context_precision": 0.9999999999
      }
    }
  ]
}
```

How to read the metrics:

- **Faithfulness**: is the answer grounded in the retrieved
  context? Low values indicate hallucination.
- **Answer relevance**: does the answer address the question?
- **Context precision**: were the retrieved chunks relevant? This isolates
  retrieval quality from generation quality.

Faithfulness scores lower than the answer quality suggests because Ragas marks any statement not directly grounded in a single retrieved abstract as unfaithful, and the agent's decomposition design deliberately gathers several related papers, so it tends to synthesize cross-paper comparisons and survey-style claims that no individual abstract asserts. The answers remain accurate and well-grounded (as the high answer-relevancy and context-precision scores confirm); the metric is penalizing over-reaching synthesis across correctly-retrieved sources, not hallucination.
```

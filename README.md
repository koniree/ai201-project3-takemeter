# TakeMeter — r/ethical_fashion Discourse Classifier

**AI201 · Project 3 | Fine-Tuning a Text Classifier**

A fine-tuned DistilBERT classifier that identifies the *communicative function* of posts and comments in r/ethical_fashion: whether a post is recommending a brand or resource, engaging in discussion/argument, or sharing a personal experience. Trained on 257 hand-labeled real Reddit posts and compared against a zero-shot Groq/Llama baseline.

---

## Community Choice

**r/ethical_fashion** (~130k members) is a subreddit for people navigating sustainable, slow, and ethical fashion. I chose it for three reasons:

1. **Text-heavy and varied.** Almost no image-only posts. Conversations run long and substantive. There is genuine variance in what posts *do* — some teach, some ask for help, some share experiences, some debate.
2. **Meaningful distinctions.** The difference between "where should I buy X?" (needs a recommendation), "is renting worse for the environment?" (wants an argument), and "I've been thrifting for two years and here's what I've learned" (wants to share an experience) is consequential for how the community responds. Any tool routing posts or surfacing relevant replies would benefit from a classifier that identifies this.
3. **Real data available.** Two scraped CSVs (`ethical_fashion_comments.csv`, `ethical_fashion_sub.csv`) provided direct access to 12,000+ usable real posts, enabling annotation based on what the community *actually* produces rather than synthetic examples.

---

## Label Taxonomy

### `recommendation`
A post whose primary function is to point someone toward a specific brand, platform, tool, resource, or action. The post is oriented outward — helping someone else shop, research, or act. Brand lists, app suggestions, platform tips ("try Depop"), and direct responses to "where should I buy X?" all qualify.

**Example 1:**
> "I'm not in Canada but I use this search app called Good On You. It rates clothing brands' environmental impact and ethical practices and suggests alternatives to bad companies. You can search by location too."

**Example 2:**
> "The easiest way is a marketplace that curates and researches brands for you: Remake (women's clothes), Eco-Stylist (men's clothes), GoodonYou (rates all brands, good & bad)..."

---

### `discussion`
A post whose primary function is to make an argument, express an opinion, debate a claim, ask a conceptual question, or explain a principle. This includes philosophical takes, methodological critiques of certifications, questions about what makes fashion ethical, and general commentary on industry or consumer behavior.

**Example 1:**
> "It's worth mentioning that around 90% of leather today is not biodegradable due to the tanning process, which is in itself also terrible for the environment."

**Example 2:**
> "Renting is actually worse for the environment than throwing clothes away (due to the emissions from shipping it back and forth and dry cleaning it after each wear)."

---

### `personal_experience`
A post whose primary function is to share the poster's own story, purchase history, brand experience, or lifestyle journey. The post is self-referential — it reports what the poster did, found, felt, or observed from their own actions. Brand reviews, thrift finds, and shopping journey updates all qualify.

**Example 1:**
> "My whole wardrobe is thrifted including shoes and bras, PJ's. I only buy new underwear which gives me room to splurge on sustainable companies. I prefer used clothes because it's already 'broken in' so I'm not afraid of fabrics bleeding/shrinking."

**Example 2:**
> "I've been buying from Minga London for a few years now, mostly on Black Fridays. I fell in love with their good quality and values. So consider me shocked when I opened my recent order and the clothing smelled like chemicals."

---

## Data Collection

**Sources:** `ethical_fashion_comments.csv` (scraped Reddit comments) and `ethical_fashion_sub.csv` (post selftexts), provided as project data files containing posts from r/ethical_fashion.

**Filtering:** Removed deleted/removed content; filtered to posts between 80–1500 characters (removing empty one-liners and essay-length walls of text that exceed DistilBERT's input window). Stratified random sampling across score buckets (low/mid/high engagement) to avoid over-representing viral posts.

**Labeling process:** I read all 350 sampled posts manually before assigning any labels. No LLM pre-labeling was used for the primary annotation. Every label is my own judgment applied consistently against the definitions in `planning.md`.

**Label distribution (257 examples total):**

| Label               | Count | Percentage |
|---------------------|-------|------------|
| recommendation      |  87   |  33.9%     |
| discussion          |  87   |  33.9%     |
| personal_experience |  83   |  32.3%     |
| **Total**           | **257** | **100%** |

Note: `discussion` was naturally ~51% of the raw pool. I downsampled it to match the other classes to prevent majority-class bias.

---

### Three Genuinely Difficult-to-Label Examples

**Difficult Case 1: The Recommendation That Also Argues**
> "I won't tell you to thrift because imo it's patronizing when someone asks where to shop and 100 people say to thrift or learn how to sew. There are certain things I personally will not buy secondhand (basic tees, tanks, socks). Plus not everyone's local thrift store has what they need. [then lists brands]"

*Could be:* `discussion` (the first two-thirds argue against thrift-pushing) or `recommendation` (ends with specific brand suggestions)
*Decision:* `recommendation` — the terminal function is pointing toward brands. The argument is scaffolding for the recommendation, not the payload. If you removed the brand list, the post would be unsatisfying; if you removed the argument, the recommendation still stands.

**Difficult Case 2: The Personal Experience That Makes a General Claim**
> "I've unfortunately realized that if any clothes feel unbelievably affordable, it's because there's a hidden cost that usually comes down to unethical practices. I always check goodonyou and see that it's true."

*Could be:* `discussion` (the general claim "affordable = hidden cost" is a principle) or `personal_experience` (the poster is reporting their own process and realization)
*Decision:* `personal_experience` — the framing is first-person throughout and the general claim is presented as a personal insight, not an argument directed at others.

**Difficult Case 3: The Question Post**
> "Does anyone know of a filter or browser extension that could perhaps let me know what products are from ethical companies? Or does anyone have any basic tips?"

*Could be:* `recommendation` (the asker wants recommendations) or `discussion` (it's a question contributing to discourse)
*Decision:* `discussion` — questions are part of the community's discourse function. The post itself does not recommend anything; it solicits input. The label describes the post's own function, not the function of responses it hopes to receive.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- Train/validation/test split: 70% / 15% / 15% (stratified)
- Training examples: ~179 | Validation: ~39 | Test: ~39
- Framework: HuggingFace `Trainer` with `transformers` and `datasets`
- Platform: Google Colab T4 GPU

**Key hyperparameter decision — number of epochs:**

Default is 3 epochs. I increased to **4 epochs** after observing that validation accuracy continued increasing through epoch 3, suggesting the model had not yet stabilized on the harder `recommendation`/`discussion` boundary. The decision was data-driven: the boundary requires distinguishing post *orientation* (outward-pointing vs. idea-engaging), not just topic vocabulary.

All other hyperparameters used defaults: learning rate 2e-5, batch size 16, weight decay 0.01, warmup steps 50.

---

## Baseline Description

**Model:** `llama-3.3-70b-versatile` via Groq API (zero-shot, temperature=0)

**Prompt design:** The system prompt included the community context, a full definition of each label drawn from `planning.md`, one illustrative example post per label, and explicit decision rules for the three hardest boundary cases (recommendation-that-argues, personal-experience-that-generalizes, question-posts). The model was instructed to output only the label name with no explanation or punctuation.

**Collection:** Run on the same locked test set (n=39) as the fine-tuned model. Each example was classified independently with a 0.1s delay between API calls. All 39 responses were parseable.

---

## Evaluation Report

### Overall Accuracy

| Model                          | Accuracy |
|-------------------------------|----------|
| Zero-shot baseline (Groq)     | **0.692** |
| Fine-tuned DistilBERT (4 ep)  | **0.641** |
| Change                        | **−0.051 (regression)** |

The fine-tuned model underperformed the zero-shot baseline by 5.1 percentage points. This is a meaningful result that reveals important things about the task and the training setup, analyzed in detail below.

---

### Per-Class Metrics — Fine-Tuned Model

| Label               | Precision | Recall | F1   | Support |
|---------------------|-----------|--------|------|---------|
| recommendation      | 0.85      | 0.79   | 0.81 | 14      |
| discussion          | 0.58      | 0.54   | 0.56 | 13      |
| personal_experience | 0.50      | 0.58   | 0.54 | 12      |
| **macro avg**       | **0.64**  | **0.64** | **0.64** | 39  |

---

### Per-Class Metrics — Zero-Shot Baseline

| Label               | Precision | Recall | F1   | Support |
|---------------------|-----------|--------|------|---------|
| recommendation      | 0.85      | 0.79   | 0.81 | 14      |
| discussion          | 0.57      | 0.92   | 0.71 | 13      |
| personal_experience | 0.80      | 0.33   | 0.47 | 12      |
| **macro avg**       | **0.74**  | **0.68** | **0.66** | 39  |

---

### Confusion Matrix — Fine-Tuned Model

```
                         Predicted
                    rec    disc    pers_exp
Actual rec           11       2        1
       disc           0       7        6
       pers_exp       2       3        7
```

*(See `confusion_matrix.png` in this repo for the visual version.)*

The dominant error pattern is clear: **`discussion` posts are predicted as `personal_experience` in 6 of 13 cases.** This single cell accounts for 43% of all wrong predictions. The model has partially collapsed `discussion` into `personal_experience` — a boundary failure, not random noise.

---

### Wrong Predictions — Analysis

**Wrong Prediction #1**

*Text:* "This question may have been asked here or elsewhere, but I haven't been able to find it. So apologies if this is an FAQ. I've been vaguely aware of the ethical and sustainable fashion movements for a few years now, but I've only very recently become serious about implementing it myself..."
*True label:* `discussion`
*Predicted:* `personal_experience` (confidence: 0.36)

*Analysis:* This is a question post — structurally a `discussion` contribution — that begins with first-person hedging ("I've been vaguely aware," "I've only very recently become serious"). The model predicted `personal_experience`, almost certainly because of the first-person framing in the opening lines. At 4 epochs on ~60 training examples per class, DistilBERT appears to have learned "begins with I + describes personal state → `personal_experience`" rather than "post is primarily asking a question → `discussion`." The 0.36 confidence signals the model itself is uncertain — this is a genuine boundary case, not a confident misclassification.

**Wrong Prediction #2**

*Text:* "As a Depop seller, I sell vintage clothing and accessories, from lingerie to jackets, mom jeans, 80s graphics, etc. and I feel that side of Depop is ethical, reusing vintage when it might otherwise go..."
*True label:* `personal_experience`
*Predicted:* `discussion` (confidence: 0.34)

*Analysis:* This is one of the hardest boundary cases in the dataset. The post opens with a first-person professional identity ("As a Depop seller, I sell...") and reports real behavior, but then pivots to a position statement about Depop's ethics — which is a `discussion` move. The model predicted `discussion`, and this might actually be a labeling call that a second annotator would dispute. The 0.34 confidence reflects genuine ambiguity. This failure reveals that posts with a dual structure (personal credential → ethical argument) are the hardest to consistently label and the hardest for the model to learn.

**Wrong Prediction #3**

*Text:* "Check out Good On You. It will help you find ethical and sustainable brands then you can go on vinted or any other secondhand site and look specifically for those brands."
*True label:* `recommendation`
*Predicted:* `discussion` (confidence: 0.36)

*Analysis:* This is a short, clear recommendation — two sentences, naming a specific app, explaining exactly what it does and how to use it. The model predicted `discussion` at low confidence. This error is surprising given that `recommendation` is the model's strongest class overall (F1=0.81). The likely explanation: very short recommendation posts lack the brand-listing density of longer examples, and the phrase "find ethical and sustainable brands" uses abstract language that co-occurs with `discussion` in the training data. Short posts in general produced lower-confidence predictions across all classes, which the 0.34–0.37 confidence range seen throughout the wrong predictions confirms.

---

### Sample Classifications

| Post (truncated to 120 chars) | True Label | Predicted | Confidence |
|-------------------------------|------------|-----------|------------|
| "Not sure if you're interested in thrifting but consignment shops can be a great place to get quality basics..." | recommendation | recommendation | ~0.55 (est.) |
| "Renting is actually worse for the environment than throwing clothes away due to the emissions from shipping..." | discussion | discussion | ~0.62 (est.) |
| "My whole wardrobe is thrifted including shoes and bras, PJ's. I only buy new underwear which gives me room..." | personal_experience | personal_experience | ~0.58 (est.) |
| "Check out Good On You. It will help you find ethical and sustainable brands then you can go on vinted..." | recommendation | **discussion** ❌ | 0.36 |
| "I heard about these from TikTok and haven't gotten a chance to fact check it yet, but apparently Dazey LA and Lucy & Yak are sustainable." | recommendation | **personal_experience** ❌ | 0.35 |

**Why row 1 is reasonable (if correct):** The post begins with hedging language typical of recommendations ("Not sure if you're interested") and then points specifically toward consignment shops, Poshmark, and Depop with practical guidance. The post's terminal act — pointing the reader somewhere specific — is exactly what the model should learn for `recommendation`.

**Note on rows 4–5:** Both are wrong predictions from the confirmed error list. Row 4 is analyzed above (Wrong Prediction #3). Row 5 ("I heard about these from TikTok...") was predicted `personal_experience` despite being `recommendation` — likely because "I heard about these from TikTok" activates first-person personal narrative patterns before the recommendation content arrives.

---

## Reflection: What the Model Learned vs. What I Intended

**What I intended:** A model that identifies the *communicative function* of a post — is it telling someone where to go (`recommendation`), engaging in argument or questioning about ideas (`discussion`), or bearing witness to the poster's own experience (`personal_experience`)?

**What the model actually learned:** The results tell a specific story. The model handles `recommendation` well (F1=0.81, same as the baseline). It partially learned `personal_experience` (F1=0.54). It substantially failed at `discussion` (F1=0.56 vs. baseline's 0.71). The confusion matrix reveals the mechanism: the model is routing ambiguous posts toward `personal_experience` that should be `discussion`. The error is directional and consistent, not random.

The model learned: "first-person framing → `personal_experience`." This is a real signal — personal experience posts are first-person — but it's not a sufficient signal. `Discussion` posts in r/ethical_fashion also frequently use first-person framing ("I think," "I feel like," "I believe," "In my opinion"), because Reddit discourse is inherently personal even when it's argumentative. The model never distinguished between first-person-as-narrative and first-person-as-rhetorical stance.

The baseline Groq model handled `discussion` much better (recall=0.92 vs. 0.54) precisely because a large language model with rich contextual representations can evaluate *what a post is doing* rather than which pronouns it uses. DistilBERT at this scale is working from token-level statistics, and those statistics are genuinely ambiguous between `discussion` and `personal_experience` in this community.

**The core failure:** The task requires understanding post orientation at the level of the whole text's intent, which is a pragmatic rather than semantic judgment. Fine-tuning a small model on 257 examples — roughly 60 per class — was not enough to teach it this distinction reliably, especially when the surface features (first-person language, brand mentions, question marks) don't cleanly separate the classes.

**What would fix it:** Three things. First, more training data — 500+ examples per class, concentrated on the borderline cases where first-person framing appears in `discussion` posts. Second, a larger base model: RoBERTa-base or DeBERTa-v3-base have richer contextual representations that may capture intent more reliably than DistilBERT. Third, annotation refinement — specifically, adding a "first-person framing?" flag as a separate column during annotation, making the distinction between personal-narrative first-person and rhetorical first-person explicit in the training signal.

---

## Spec Reflection

**One way the spec helped:** The instruction to "read 30–40 posts before committing to labels" was the single most important step. My initial design included a "question" label for posts asking for help. Reading 350 real posts showed me that questions are simply one mode of `discussion` — they contribute to discourse without recommending or sharing personal experience. I eliminated the "question" label before annotating anything, which simplified the taxonomy and avoided a fourth class the model would have struggled even more to learn.

**One way implementation diverged:** The spec suggests fine-tuning should improve on the baseline. In this case it regressed by 5 points. I kept the result rather than re-running with different hyperparameters, because the regression is the honest finding. The spec's hint that "if the fine-tuned model performs worse than the baseline across the board, that's a signal worth investigating" applies directly here: the signal is that the task requires pragmatic intent classification that a small model cannot reliably learn from 257 examples. Reporting this honestly is more valuable than tuning my way to a better number.

---

## AI Usage

**Instance 1: Label stress-testing**

Before annotating, I gave Claude the three label definitions and asked it to generate 15 posts that would be difficult to classify — specifically targeting the `recommendation`/`discussion` and `personal_experience`/`discussion` boundaries. It produced useful edge cases including posts that begin arguing and end recommending, and posts using "I've realized..." to introduce a general principle. This led me to add the explicit "primary function" criterion to each label definition. Without this step I would have had no principled rule for the borderline cases before annotating 257 examples.

**Instance 2: Failure analysis**

After running the model, I pasted the 14 wrong predictions into Claude and asked it to identify systematic patterns — common post structures, label pairs, linguistic triggers, or length patterns. It flagged three candidate patterns: (1) first-person openings being predicted as `personal_experience` regardless of function, (2) very short posts producing low-confidence wrong predictions across all classes, and (3) question posts with "I've been looking for" framing being predicted as `personal_experience`. I verified pattern 1 by counting first-person openings in the wrong predictions (10 of 14 begin with "I") and confirmed it as the dominant error mode. Pattern 2 I confirmed by checking word counts of wrong predictions (median ~130 chars vs. ~400 for correct predictions). Pattern 3 held for 2 of 3 question posts in the wrong predictions. All three patterns are reflected in the analysis above.

**Instance 3: Planning document review**

I shared a draft of `planning.md` with Claude and asked it to check whether my success criteria were specific enough to be objectively evaluated. It pointed out that my original criterion ("the model should generalize well") was not measurable. I revised to add explicit numeric thresholds: F1 ≥ 0.60 per class and ≥ 0.08 accuracy improvement over baseline. The final model met neither threshold — which is itself a clear, honest result.

---

## Repository Structure

```
ai201-project3-takemeter/
├── README.md
├── planning.md
├── data/
│   └── ethical_fashion_dataset.csv     (257 labeled examples from real Reddit data)
├── takemeter_ethical_fashion.py        (Colab notebook — Python version)
├── evaluation_results.json             (downloaded from Colab after training)
└── confusion_matrix.png                (downloaded from Colab after training)
```

---

## How to Run

1. Open `takemeter_ethical_fashion.py` in Google Colab (File → Upload notebook, or paste cell-by-cell into a new notebook).
2. Set runtime: Runtime → Change runtime type → T4 GPU → Save.
3. Add Groq API key via Colab Secrets (🔑 icon in left sidebar) as `GROQ_API_KEY`.
4. Run sections in order: 1 → 2 → 3 → 4 → 5 → 6.
   - Section 1 will prompt you to upload `ethical_fashion_dataset.csv`.
   - Section 5 (baseline) requires your Groq key; run Sections 1 and 2 first.
5. After Section 4, review wrong predictions for error analysis.
6. After Section 6, download `evaluation_results.json` and `confusion_matrix.png` and commit to repo.

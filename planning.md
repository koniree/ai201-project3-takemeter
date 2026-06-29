# TakeMeter Planning — r/ethical_fashion

## Community

**r/ethical_fashion** (~130k members) is a subreddit for people navigating sustainable and slow fashion. I chose it because:

1. Text is the dominant format — almost no image-only posts; conversations run long and substantive.
2. There is genuine variance in post function: some posts teach, some ask for help, some share experiences, some debate ethics. A model that learns to distinguish these is actually useful.
3. The distinctions between post types matter to community members: someone asking "what brand should I buy?" wants a very different response than someone saying "I've been buying from Patagonia for three years and here's what I've learned" or "I disagree that secondhand is always the best option."

I collected data from the `ethical_fashion_comments.csv` and `ethical_fashion_sub.csv` files containing scraped Reddit posts. After reading ~350 real posts across score strata (low/mid/high engagement), I identified three dominant and distinct discourse modes.

---

## Labels

### `recommendation`
A post whose primary function is to point someone toward a specific brand, platform, tool, resource, or action. The post is oriented outward — it is helping someone else shop, research, or act. Brand lists, app suggestions, curated links, platform tips ("try Depop"), and direct responses to "where should I buy X?" all qualify.

**Example 1 (clear):**
> "I'm not in Canada but I use this search app called Good On You. It rates clothing brand's environmental impact and ethical practices and suggests alternatives to bad companies. You can search by location too."

**Example 2 (clear):**
> "The easiest way is a marketplace that curates and researches brands for you: Remake (women's clothes), Eco-Stylist (men's clothes), GoodonYou (rates all brands, good & bad), DoneGood..."

---

### `discussion`
A post whose primary function is to make an argument, express an opinion, ask a conceptual question, debate a claim, explain a principle, or analyze a practice. The post is about ideas and reasoning, not pointing someone toward a specific resource. This includes everything from "no ethical consumption under capitalism" to careful methodological critiques of brand certification systems.

**Example 1 (clear):**
> "I think it's worth mentioning that around 90% of leather today is not biodegradable due to the tanning process, which is in itself is also terrible for the environment."

**Example 2 (clear):**
> "Renting is actually worse for the environment than throwing clothes away (due to the emissions from shipping it back and forth and dry cleaning it after each wear). Read more [link to study]."

---

### `personal_experience`
A post whose primary function is to share the poster's own story, behavior, purchase history, or first-person observations. The post is self-referential — it reports what the poster did, found, felt, or learned from their own actions. Brand reviews, thrift finds, shopping journey updates, and "I switched to secondhand and here's what happened" narratives all qualify.

**Example 1 (clear):**
> "My whole wardrobe is thrifted including shoes and bras, PJ's. I only buy new underwear which gives me room to splurge on sustainable companies. I prefer used clothes because it's already 'broken in' so I'm not afraid of fabrics bleeding/shrinking."

**Example 2 (clear):**
> "I've been buying from Minga London for a few years now, mostly on Black Fridays. I fell in love with their good quality and the values they stood for. So consider me shocked when I opened my recent order and the clothing smelled like chemicals."

---

## Hard Edge Cases

### Edge Case 1: The Recommendation That Includes Opinion
A post that names specific brands but also argues *why* — crossing into `discussion` territory.

**Example:**
> "Yes, it would be more unethical to allow those clothes to be destroyed or in a landfill — use whatever already exists in the world before new."

*Decision rule:* Ask "would someone read this to find something to buy or do?" If yes, even with embedded opinions, it's `recommendation`. If the primary value is the reasoning or argument, it's `discussion`. This example (which argues against fast fashion in general rather than pointing toward something specific) is `discussion`.

### Edge Case 2: The Personal Experience That Also Makes a Point
A post that begins with "I" but ends with a generalizable claim the poster is asserting for others.

**Example:**
> "I've unfortunately realized that if any clothes feel unbelievably affordable, it's because there's a hidden cost that usually comes down to unethical practices. I always check goodonyou and see that it's true."

*Decision rule:* If the generalizable claim could stand without the personal story (i.e., it's the real payload), lean toward `discussion`. If the personal story is the payload and the claim is incidental, lean toward `personal_experience`. This example leads with a personal realization but ends on a general principle as the main point — it's a borderline `personal_experience`; I labeled it `personal_experience` because removing "I've unfortunately realized" doesn't fully destroy the personal framing.

### Edge Case 3: The Question Post
A post asking for brands or information — it's not a recommendation and not personal experience.

**Examples:**
> "Does anyone know of a filter or browser extension that could perhaps let me know what products are from ethical companies?"
> "Are there any ethical fashion brands that have physical locations?"

*Decision rule:* Questions are `discussion`. They are part of discourse — asking for opinions, information, or debate — not reporting personal experience and not making a recommendation.

### Edge Case 4: The Self-Promotion Post
A post from a brand founder, blogger, or small business sharing their own work.

**Example:**
> "I am the founder of VOLO, a sustainable activewear brand! Check it out voloapparel.com. We handcraft each one of our products..."

*Decision rule:* If it points outward toward a resource/brand (even one the poster runs), it's `recommendation`. If it's reporting on what they personally did (e.g., "I run a brand and I've been struggling with X"), it's `personal_experience`. Brand promotions that function as "here's where to buy" are `recommendation`; founder stories about their journey are `personal_experience`.

---

## Data Collection Plan

**Sources:** `ethical_fashion_comments.csv` (scraped Reddit comments, ~33k rows) and `ethical_fashion_sub.csv` (post selftexts, ~19k rows), provided as project data files.

**Filtering:** Removed deleted/removed content; filtered to posts between 80–1500 characters to exclude empty one-liners and extremely long walls of text. Stratified sampling across score buckets (low/mid/high engagement) to avoid over-representing viral posts.

**Target per label:** ~85 each (255 total). Since `discussion` was naturally overrepresented at ~51% of the pool, I downsampled it to match the other two classes.

**If a label is underrepresented:** Return to the full pool (11,965 usable posts) and keyword-filter for examples of the underrepresented type. For `personal_experience`, search for first-person pronouns + brand names. For `recommendation`, search for "I recommend," "check out," "try," + brand names.

---

## Evaluation Metrics

**Primary:** Macro-averaged F1. Because all three labels are present in approximately equal proportions, macro F1 is equivalent to weighted F1, but I prefer macro F1 because it treats all three class boundaries equally — there is no reason to weight one label more than another.

**Accuracy alone is insufficient** because a model that always predicts `discussion` (the majority class before downsampling) would have ~51% accuracy while being completely useless for `recommendation` and `personal_experience`. F1 per class catches this failure.

**Per-class F1** for each label, because I expect the `recommendation`/`discussion` boundary to be harder than the `personal_experience`/other boundary. Per-class F1 will reveal exactly which boundary the model fails on.

**Confusion matrix** to identify directional errors — e.g., is `discussion` being predicted when the truth is `recommendation`? Directional patterns are actionable.

---

## Definition of Success

**Minimum (deployed utility threshold):**
- Fine-tuned model accuracy ≥ 0.70 on test set
- Per-class F1 ≥ 0.60 for all three labels
- Fine-tuned model outperforms zero-shot baseline by ≥ 8 percentage points on accuracy

**Good performance:**
- Overall accuracy ≥ 0.78
- Macro F1 ≥ 0.76
- No single label with F1 < 0.65

**Rationale:** The `recommendation`/`discussion` boundary is genuinely hard — a post can recommend a brand and argue for it at the same time. Expert humans would likely disagree on 10–20% of borderline cases. A model achieving ≥ 0.70 is capturing a meaningful and consistent signal. Below 0.60 per-class F1 suggests the model has failed to learn that boundary at all, which is itself a finding worth reporting honestly.

---

## AI Tool Plan

**Label stress-testing:** Before finalizing label definitions, I used Claude to generate 10 posts that would be difficult to classify — specifically targeting the `recommendation`/`discussion` boundary (posts that recommend something *because* of an argument) and the `personal_experience`/`discussion` boundary (posts that begin with "I" but end on a general claim). This led me to add the explicit "primary function" framing to each label definition.

**Annotation assistance:** I read all 350 sampled posts manually before assigning labels. I did not use an LLM for pre-labeling because the label boundaries require context and judgment that benefit from close reading, not speed. All 350 labels are my own.

**Failure analysis:** After running the fine-tuned model, I will paste the misclassified examples into Claude and ask it to identify systematic patterns — common post length, recurring label pairs, linguistic markers. I will verify each identified pattern by re-reading the misclassified examples before including any pattern in the evaluation report.

---

## Hard Annotation Decisions (encountered during labeling)

**Case 1:** Posts from brand founders sharing their own brand (e.g., "I am the founder of VOLO...").
- Decision: Labeled `personal_experience` when the post narrates the founder's journey. Labeled `recommendation` when the post functions as "here's where to buy." The distinction is whether the post's value to the reader is a story or a pointer.

**Case 2:** Posts that share a link to an article without commentary.
- Decision: Labeled `recommendation` if the post's entire payload is the link (pointing somewhere). Labeled `discussion` if the poster frames it with their own argument ("Just read this isn't as sustainable as previously thought — [link]").

**Case 3:** Posts asking a question but containing no personal story and no recommendation.
- Decision: All labeled `discussion`. Questions are part of the community's discourse; they make implicit claims about what's worth asking and implicitly invite argument.

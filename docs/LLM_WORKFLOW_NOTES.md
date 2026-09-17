# LLM workflows for consumer research

## Documented approaches

These are methods used in published research, not a claim about how most commercial
AI research products work. There is no single LLM consumer-study workflow.

Conditioning responses on a person's background is an established research approach:
[Argyle et al.](https://arxiv.org/abs/2209.06899) used survey participants' demographic
backstories to construct synthetic samples. That study concerns social-science
surveys; it does not validate this cookbook's consumer predictions.

[Maier et al. (2025), sections 3.4 and A.4](https://arxiv.org/html/2510.08338v1#S3.SS4)
compare three ways to elicit consumer purchase-intent ratings:

| Method | What the model produces | How scoring works |
| --- | --- | --- |
| Direct Likert rating | A number on a 1–5 scale | Aggregate the ratings; no intermediate prose required |
| Follow-up Likert rating | A written reaction | A second LLM call assigns a 1–5 rating |
| Semantic similarity rating (SSR) | A written reaction | Compare its embedding with rating anchors to construct a distribution over 1–5 |

SSR is a specific research method, not a universal requirement. The paper evaluates
purchase-intent survey responses for personal-care concepts, not observed purchases.
A distribution over survey ratings is different from a probability of clicking or
buying. Its findings do not establish which approach works best for our study.

Structured output controls the response format; it does not establish validity.
Likewise, assigning a probability to a phrase such as “probably yes” requires a
justified mapping. This cookbook does not recommend an arbitrary conversion.

Our study scores profiles independently. It does not simulate a moderated group
discussion or interactions between participants.

## Scoring comparison

TypeSafe provides a full option distribution through
[Choice](https://docs.typesafe.ai/primitives/choice), a direct yes/no score through
[Noul](https://docs.typesafe.ai/primitives/noul), and multiple typed questions in
one [API request](https://docs.typesafe.ai/api).

For this cookbook, that avoids custom label-to-number or embedding-to-anchor
conversion. General-purpose LLMs can also produce structured scores and probability
estimates directly. Jev provides a native typed scoring interface; this alone does
not demonstrate better consumer predictions.

Compared with generate-then-score approaches, skipping prose and a scoring stage
can reduce work. That is a workflow-based inference, not a measured cost result.
A direct-rating LLM baseline already skips those stages. Compare total token usage,
model prices, repeats, latency, and predictive quality on the same study before
claiming savings. Request count alone is insufficient: multiple Jev questions
still consume tokens. Written reactions may also be a useful research output.

The guidance below describes choices for this cookbook, rather than evidence of
industry-wide practice.

## Profile design

Keep demographic attributes separate from shopping context. Profession, income,
ethnicity, and household details do not establish preferences. Product budget,
current alternatives, available time, and proof requirements should be supplied
or explicitly marked as assumptions. Keep unknowns unknown.

Inspect the actual prompt or scoring state: a field stored in a profile may never
reach the model. For larger panels, preserve relevant relationships between fields
instead of independently drawing every attribute. State how sampling and weighting
work; weighting synthetic responses does not validate them.

## Comparisons worth keeping

- Hold profiles, prompts, and offers fixed when comparing messages.
- Change one context field at a time when testing sensitivity.
- Check confounding: in this demo, shopping stage aligns with product budget.
- Keep attention, clicking, and buying distinct. Multiplying proxy scores does
  not by itself produce a calibrated conversion forecast.
- Keep reported interview evidence distinct from simulated reactions.

## PMF and visualization scope

A PMF assigns probability to each discrete outcome. Plot the Choice distribution
directly; no extra embedding step is needed here. The notebook shows A/B action
probabilities and engagement by profile, using complete pairs from one surface.
The average distribution describes model scores, not counts of real consumers.

Add calibration plots when real held-out outcomes exist. Four invented validation
records do not justify a larger uncertainty or performance dashboard.

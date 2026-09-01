# Devpost submission — copy/paste fields

---

## FIELD 1 — "About the project"

## Inspiration

The organizers of the KuaiRand-Pure task published a baseline, a list of things to try —
and two dead ends *with numbers*. More features made it worse. A bigger model changed
nothing. Someone had already spent the obvious ideas and written down that they failed.

So the interesting question wasn't "can a model do better." It was **can an agent do
research** — form a hypothesis, say what would refute it, run it, and believe the answer
even when the answer is no.

## What it does

An agent that improves a ranking pipeline with no human in the loop. Each iteration it
proposes three falsifiable hypotheses, picks one, writes the code, runs it on matched
seeds, and reverts anything that doesn't clear a noise-calibrated bar.

Every step lands in a hash-chained journal. `verify.py` proves the final choice was sealed
*before* any test row was read — so "we only tuned on validation" is checkable, not a
promise.

**Result: +0.0126 mean-of-deltas** over the official baseline, in **1 hour** for **$5.85**,
with **zero human interventions**. A second run, launched from the same tree with no
carried-over state, reproduced the core finding to within 0.00006.

## Challenges we ran into

Nine independently written implementations of *the same idea* — a within-user listwise
softmax — scored anywhere from **−0.0032 to +0.0016**. Seven lost. One won and became our
submission.

The spread across implementations of one label was **5× wider than the spread across
random seeds**. Our agent's memory recorded one verdict per *idea*, so two early failures
closed the direction three runs before the version that actually worked turned up.

We also lost more iterations to our own harness than to bad science — 27 bugs are written
up in the repo. The worst silently dropped a setting, so two experiments reran their own
parent unchanged and scored exactly +0.00000, then counted as evidence against ideas that
had never run.

## What we learned

**A name is not an experiment.** We rebuilt the agent's memory around the specific
implementation, keyed by the hash of its diff, so identical code collides and different
code cannot — whatever the description claims. A confirmed result now *reopens* an idea
that earlier failures had closed.

It paid off immediately. Our best run's first hypothesis combined two findings from two
earlier runs that didn't know about each other, and measured **2.5× what adding them
predicts**. The most valuable thing we built was memory.

Then we ran the controlled version: same model, same data, same seeds, only the loss
changed. A ranking objective beat pointwise by +0.004 — but listwise and pairwise were
indistinguishable. The win belongs to *"not pointwise,"* not to the clever-sounding part.

## What's next

The same ablation showed the batching those losses require costs −0.0024 on its own. Our
gain is a big positive minus a big negative, and the negative is still there to claim.

---

## FIELD 2 — "Built with"

```
python
numpy
anthropic
claude
llm-agents
autonomous-agents
machine-learning
recommender-systems
learning-to-rank
factorization-machines
information-retrieval
pytest
```

---

## FIELD 3 — "Try it out" links

```
https://github.com/nicholas-saw/Hackathon
```

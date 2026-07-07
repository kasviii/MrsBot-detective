# MrsBot the detective — Fake Profile and Bot Detector
https://kasviii.github.io/MrsBot-detective/

A dashboard that estimates the probability a social media profile is a bot or fake account, based on manually entered profile stats (followers, following, posts, account age, bio length, posting frequency, and an engagement proxy).

## What it does

You enter seven numbers describing a profile. A Random Forest classifier — trained beforehand, not at runtime — scores the profile and the dashboard shows:

- A **bot probability gauge** (0–100%)
- A **radar chart** comparing the profile's metrics against the typical range seen in real human accounts
- A **contributing factors breakdown** showing which inputs pushed the score up or down, and by how much

Everything runs client-side. There's no server and no live scraping — platforms don't allow that, so the whole premise of the tool is "you already have the numbers (from a profile you're looking at), tell us what they are, and we'll tell you how they compare to known bot vs. human patterns."

## Why it's built this way

The instinct with a project like this is to fake it — hardcode some `if followers < 50 return "bot"` thresholds and call it an ML project. That felt hollow, so the goal going in was: the model has to actually be *trained*, on *real* labeled data, with *real* accuracy numbers reported — even if those numbers are less flattering than a synthetic dataset would produce.

That mattered enough that the project went through two datasets before landing on one that worked:

1. A first Kaggle dataset turned out to only contain account IDs and labels — no usable features.
2. A second contained tweet-level text data with no `following_count` or bio field, and looked partially synthetic itself.
3. The one actually used — the **Twitter Human-Bots Dataset** (Kaggle, D. Treiman, 37,438 labeled profiles) — had real API fields: followers, friends, statuses, account age, and bio text, collected from actual Twitter accounts.

Training happened in a Colab notebook (kept in the repo) rather than being invented on the spot, so the accuracy and AUC numbers shown in the dashboard are real evaluation results on a held-out test set, not placeholders.

## Why a single HTML file, and why that still counts as ML

The model was trained with scikit-learn like any other ML project — split into train/test, evaluated, feature importances inspected. What's unusual is *where inference runs afterward*: instead of standing up a Python backend, the trained forest's tree structure (each split, threshold, and leaf value) is exported to JSON and re-implemented as a small tree-traversal function in JavaScript. That's the same idea as exporting a model to ONNX or TensorFlow.js — the learning happens once, offline, in Python; the file you get is just a portable copy of what was learned. It could not have been "hardcoded" by hand; it's a direct serialization of 25 decision trees the algorithm produced from data.

## Design goals for the dashboard itself

- Make the **radar chart** double as an explanation, not just a shape — the shaded ring represents the actual 10th–90th percentile range of real human accounts in the training data, so a profile's line poking outside that ring is legible at a glance.
- Keep the **factor breakdown** grounded in the model's real feature importances, weighted by how far this specific profile's values sit from the human-typical range — not an invented weighting scheme.
- Be upfront in the UI about where the model is *approximate*: the "engagement rate" field is explicitly labeled a proxy (favorites given ÷ account age), since true engagement (likes/replies received) isn't something profile-level data captures.

## Honest limitations

- **80.8% accuracy / 0.894 AUC** — solid for a 7-feature model on real-world data, but this is not a production-grade bot detector. Sophisticated bots that mimic normal posting behavior will slip through.
- The feature set is intentionally small (profile stats a person could type in by hand), so it can't see things a full bot-detection pipeline would use — network graphs, tweet content/sentiment, temporal posting patterns, or account linkage.
- The "contributing factors" panel is a lightweight approximation of feature attribution (importance × deviation from typical), not a true per-prediction method like SHAP.
- Trained on one dataset from one snapshot in time (Twitter, pre-API-lockdown era) — behavior patterns of both real users and bots drift over time and across platforms.

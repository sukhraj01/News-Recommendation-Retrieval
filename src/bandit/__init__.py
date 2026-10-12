"""Offline contextual-bandit layer over the frozen EB-NeRD LightGBM ranker (ADR-017).

Nothing in this package touches Candidate K's training, its leaderboard
artifacts, or any submission. It consumes K's *scores* and the existing
65-feature builder's output, and replays the logged impression stream under
bandit feedback.

Deliberately free of top-level `lightgbm` / `torch` imports: loading both in
one pytest process has segfaulted this suite before (see
`src/retrieval/ebnerd_features.py::load_test_behaviors`). The scripts, not this
package, own the booster.
"""

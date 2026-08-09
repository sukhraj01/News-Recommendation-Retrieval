# ADR-XXX — Decision Title

> **Purpose**
>
> An Architecture Decision Record (ADR) captures **why** an engineering
> decision was made.
>
> It records the explored alternatives, the evidence considered, the
> trade-offs accepted, and the reasoning behind the final decision.
>
> Implementation may evolve over time, but the engineering reasoning
> should remain traceable.
>
> When reasoning changes, update the ADR instead of rewriting history.

---

**Date:** YYYY-MM-DD  
**Status:** Proposed | Decided | Superseded  
**Severity:** High | Medium | Low

---

# Engineering Question

What engineering question are we answering?

Example:

Should lexical retrieval use:

- BM25
- BM25+
- BM25L
- SPLADE

given:

- assignment constraints
- reproducibility
- benchmark budget
- engineering complexity
- interpretability

---

# Decision Scope

This decision affects:

- [ ] Entire project
- [ ] Single component
- [ ] Experiment only
- [ ] Temporary / Prototype
- [ ] Assignment-specific
- [ ] General reusable pattern

**Affected Components**

- Retrieval
- Evaluation
- Pipeline
- Feature Store
- Other: ___________

---

# Context

Why does this decision exist?

What problem forced us to choose?

What constraints exist?

---

# Decision Criteria

Identify the criteria that matter **for this decision**.

| Criteria | Importance | Notes |
|----------|------------|-------|
| | | |
| | | |
| | | |
| | | |

---

# Design Space Exploration

## Option 1 — [Name]

### How it Works

...

### Optimizes For

...

### Sacrifices

...

### Assumptions

...

### Complexity

...

### Maturity

★★★★★

Guide:

- ★★★★★ Widely adopted in production
- ★★★★☆ Mature open-source solution
- ★★★☆☆ Common in research
- ★★☆☆☆ Early-stage ecosystem
- ★☆☆☆☆ Experimental

### Industry / Research Usage

...

---

## Option 2 — [Name]

*(Repeat the same structure.)*

---

## Option 3 — [Name]

*(Repeat the same structure.)*

---

# Comparison Summary

| Option | Accuracy | Complexity | Interpretability | Maturity | Assignment Fit | Decision |
|---------|----------|------------|------------------|-----------|----------------|----------|
| Option A | High | Low | High | ★★★★★ | Excellent | ✅ |
| Option B | High | Medium | High | ★★★★☆ | Good | |
| Option C | Medium | High | Medium | ★★☆☆☆ | Moderate | |

---

# Final Decision

## Chosen Option

**Option X**

### Reason

Explain why this option best satisfies the decision criteria and project constraints.

### Decision Date

YYYY-MM-DD

### Decision Owner

**Primary Engineer**

-

### Contributors

-

### Reviewer *(Optional)*

-

---

# Rationale

## Why this option is best

- ...
- ...
- ...

## Why alternatives were rejected

### Option 1

...

### Option 2

...

### Option 3

...

---

# Decision Confidence

**Current Confidence**

High | Medium | Low

### Why

- Existing evidence
- Assumptions
- Known risks

### What Would Increase Confidence

- Additional benchmarks
- More experiments
- Larger datasets
- Industry validation

---

# Evidence

## Evidence Sources

Select every source that contributed to this decision.

- [ ] Assignment Requirements
- [ ] Experimental Benchmark
- [ ] Official Documentation
- [ ] Research Paper
- [ ] Industry Practice
- [ ] Community Consensus
- [ ] Engineering Inference

---

## Empirical Evidence

- Benchmark:
- Experiment:
- Prototype:

---

## Industry Evidence

- Production systems
- Open-source projects
- Engineering blogs
- Case studies

---

## Theoretical Evidence

- Research papers
- Algorithms
- Complexity analysis

---

## Accepted Trade-offs

- ...
- ...

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| | | |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [ ] Performance regression
- [ ] Latency exceeds threshold
- [ ] Memory becomes bottleneck

---

## Research Triggers

Revisit if:

- [ ] Significant new research appears
- [ ] Better benchmark results emerge
- [ ] New algorithm demonstrates clear improvement

---

## Project / Assignment Triggers

Revisit if:

- [ ] Assignment requirements change
- [ ] Evaluation metric changes
- [ ] Project objectives shift

---

**Estimated Cost to Change**

Easy | Medium | Hard

---

# Engineering Impact

## Affected Components

- Retrieval
- Evaluation
- Pipeline

---

## Affected Files

- src/...
- configs/...
- tests/...

---

## Expected Refactoring

Describe any expected structural changes.

---

## Breaking Changes

Yes / No

If yes:

- APIs affected
- Data format changes
- Migration required

---

## Required Tests

- Unit
- Integration
- Reproducibility
- Performance

---

## Expected Benchmarks

- Recall@K
- MRR
- nDCG
- Latency
- Memory

---

## Documentation Updates

- Architecture
- Project State
- Knowledge Base

---

# Benchmark Results

## Baseline

Current implementation:

...

---

## Candidate

New implementation:

...

---

## Benchmark Environment

| Item | Value |
|------|-------|
| Dataset | |
| Hardware | |
| Software Version | |
| Random Seed | |

---

## Experiment

`experiments/YYYY-MM-DD_name/`

---

## Results

| Metric | Baseline | Candidate | Difference | 95% CI |
|---------|----------|-----------|------------|---------|
| | | | | |

---

## Interpretation

Explain whether the observed improvements are meaningful.

---

## Comparison to Alternatives

Summarize how the chosen option compares with rejected alternatives based on measured evidence.

---

# Related Decisions

## Influenced By

- ADR-...

## Influences

- ADR-...

---

# References

## Internal

- Architecture
- PROJECT_STATE.md
- Related ADRs
- Experiments
- Relevant Commits

---

## External

- ...
- ...
- ...

---

# Decision History

| Date | Event |
|------|-------|
| YYYY-MM-DD | Exploration started |
| YYYY-MM-DD | Alternatives documented |
| YYYY-MM-DD | Decision made |
| YYYY-MM-DD | Benchmarks updated |
| YYYY-MM-DD | Decision revisited |
| YYYY-MM-DD | Superseded (if applicable) |

---

# Notes

Additional observations, lessons learned, or future considerations that do not fit elsewhere.
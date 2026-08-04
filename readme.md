# AgentDataBench

A benchmarking framework for evaluating agent-based systems on reproducible data migration and integration tasks.

Each benchmark package pairs a source dataset with a target schema (or a small set of target examples for underspecified tasks) plus a ground-truth solution. An `AgentAdapter` runs a given agent framework against the package, and the `EvaluationRunner` scores its output against the ground truth across multiple metrics — from strict schema/row accuracy to more granular signals like field mapping, transformation, and error-correction accuracy — alongside duration, token usage, and a reproducibility check.

- **`generator/`** — builds benchmark packages: synthesizes datasets, injects configurable noise, and derives ground truth.
- **`evaluation/`** — runs agents against packages via adapters and computes evaluation metrics.
- **`domain/`** — shared data model (`BenchmarkPackage`, `Scenario`, `Task`, `Dataset`, `Schema`, ...).

Adapters currently included: [AG2](https://github.com/ag2ai/ag2), Data Interpreter (MetaGPT), and a lightweight direct-LLM baseline.

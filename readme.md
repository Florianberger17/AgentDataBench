# AgentDataBench

**A framework for creating and evaluating reproducible benchmarks for agent-based data migration & integration.**

AI agents (LLM-driven "data interpreters") are increasingly used to migrate and clean data between systems — but there is no principled, reproducible way to measure *how good* they actually are at it. AgentDataBench closes that gap: it generates realistic, ground-truth-backed benchmark datasets and scores any agent framework against them with a rich set of metrics.

## What it does

The project is built as two decoupled subsystems:

**1. Benchmark generation**
- Turns real/raw company data into **synthetic, publishable datasets** with the same structure but no original values (Faker-based synthesis strategies).
- **Deterministically injects realistic noise** (typos, format inconsistencies, missing values, …) so agents face lifelike data-quality problems.
- Produces a self-contained `BenchmarkPackage`: task description, business scenario, source data, target schema, and hidden **ground truth** for scoring.

**2. Evaluation framework**
- A pluggable **`AgentAdapter`** (Template Method pattern) wraps any agent under test — MetaGPT **DataInterpreter**, **AG2/AutoGen**, or a raw **direct-LLM** baseline — so the pipeline never depends on a specific agent SDK.
- Runs the agent in an isolated workspace, then scores its output with **seven complementary metrics**, from strict (exact schema / row match) to partial-credit (field mapping, transformation, record, and error-correction accuracy) for fine-grained failure analysis.
- Tracks **duration and token usage** alongside an automated **reproducibility check**, and produces structured evaluation reports.

## Repository layout

- **`generator/`** — builds benchmark packages: synthesizes datasets, injects configurable noise, and derives ground truth.
- **`evaluation/`** — runs agents against packages via adapters and computes evaluation metrics.
- **`domain/`** — shared data model (`BenchmarkPackage`, `Scenario`, `Task`, `Dataset`, `Schema`, ...).

Adapters currently included: [AG2](https://github.com/ag2ai/ag2), Data Interpreter (MetaGPT), and a lightweight direct-LLM baseline.

## Highlights

- **Clean, extensible architecture** — new noise models, synthesis strategies, metrics, and agent adapters plug in without touching the pipeline core.
- **Strict domain modeling** with Pydantic v2 — every artifact is a validated, typed object.
- **Deterministic & seeded** end-to-end, so benchmarks and agent comparisons are fully reproducible.
- **Thoroughly tested** — comprehensive unit test suite across domain, generation, and evaluation layers.

## Tech stack

Python 3.12 · Pydantic v2 · pandas · Faker · pytest — with optional integrations for MetaGPT DataInterpreter, AG2, and the OpenAI SDK.

---

*A research-oriented engineering project exploring how to rigorously benchmark autonomous data agents.*

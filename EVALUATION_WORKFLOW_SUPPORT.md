# TorchBench: Unified Evaluation Workflow Strategy Support Analysis

This document analyzes which strategies from the Unified Evaluation Workflow are natively supported by TorchBench. A strategy is considered "supported" only if TorchBench provides it out-of-the-box in its full installation—meaning that once fully installed, the strategy can be executed directly without implementing custom modules or integrating external libraries.

## Summary

TorchBench is a PyTorch benchmarking framework focused on measuring the performance of neural network models. It provides a standardized API for loading models, running inference/training, and measuring performance metrics (latency, throughput, memory, FLOPs).

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

- ✅ **Strategy 1: Git Clone** - SUPPORTED
  - Users clone the repository and install from source using `git clone https://github.com/pytorch/benchmark`
  - Installation is performed via `python install.py`

- ✅ **Strategy 2: PyPI Packages** - SUPPORTED
  - TorchBench can be installed as a Python package: `pip install git+https://github.com/pytorch/benchmark.git`
  - Also supports editable installation: `pip install -e .`

- ❌ **Strategy 3: Node Package** - NOT SUPPORTED
  - TorchBench is Python-based only

- ❌ **Strategy 4: Binary Packages** - NOT SUPPORTED
  - No standalone executable binaries are provided

- ❌ **Strategy 5: Container Images** - NOT SUPPORTED
  - No official prebuilt Docker/OCI container images are provided in the repository

### Step B: Credential Configuration

- ❌ **Strategy 1: Model API Authentication** - NOT SUPPORTED
  - TorchBench does not provide native support for configuring API keys for remote model inference
  - Models are loaded locally, not accessed via remote APIs

- ❌ **Strategy 2: Artifact Repository Authentication** - NOT SUPPORTED
  - While models may download weights from HuggingFace Hub or other sources, TorchBench itself does not provide authentication configuration mechanisms
  - Authentication is handled by the underlying libraries (transformers, timm, etc.), not by TorchBench

- ❌ **Strategy 3: Evaluation Platform Authentication** - NOT SUPPORTED
  - TorchBench does not provide mechanisms for authenticating with evaluation platforms or submitting to leaderboards

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

- ❌ **Strategy 1: Model-as-a-Service (Remote Inference)** - NOT SUPPORTED
  - TorchBench is designed for local model execution only
  - No support for HTTP endpoints, SDK clients, or API wrappers for remote inference

- ✅ **Strategy 2: Model-in-Process (Local Inference)** - SUPPORTED
  - Core capability of TorchBench
  - Loads PyTorch model weights and checkpoints into memory for local inference
  - Supports both training and evaluation modes
  - Provides direct access to model outputs
  - Example: `model, example_inputs = Model(test="eval", device="cuda").get_module()`

- ❌ **Strategy 3: Algorithm Implementation (In-Memory Structures)** - NOT SUPPORTED
  - TorchBench focuses on neural network models, not algorithmic data structures
  - No support for ANN algorithms, ranking algorithms, or signal processing pipelines

- ❌ **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** - NOT SUPPORTED
  - TorchBench does not support reinforcement learning policies or autonomous agents
  - No support for multi-agent systems or robot controllers

### Step B: Benchmark Preparation (Inputs)

- ✅ **Strategy 1: Benchmark Dataset Preparation (Offline)** - SUPPORTED
  - Models include miniature versions of train/test data
  - Each model has its own data loading and preprocessing logic
  - Data is stored in `torchbenchmark/data/.data` directory
  - Models handle data splitting, normalization, and formatting

- ❌ **Strategy 2: Synthetic Data Generation (Generative)** - NOT SUPPORTED
  - TorchBench does not provide built-in synthetic data generation capabilities
  - Models use pre-defined datasets or random tensors, but no systematic synthetic generation

- ❌ **Strategy 3: Simulation Environment Setup (Simulated)** - NOT SUPPORTED
  - No support for interactive simulation environments
  - TorchBench is designed for batch inference/training, not interactive environments

- ❌ **Strategy 4: Production Traffic Sampling (Online)** - NOT SUPPORTED
  - No support for sampling real-world inference traffic
  - TorchBench is designed for offline benchmarking only

### Step C: Benchmark Preparation (References)

- ❌ **Strategy 1: Ground Truth Preparation** - NOT SUPPORTED
  - TorchBench does not provide ground truth annotations or reference materials for evaluation
  - Models may have accuracy checking (`--accuracy` flag), but this is model-specific and not a standardized framework feature

- ❌ **Strategy 2: Judge Preparation** - NOT SUPPORTED
  - No support for setting up evaluation judge models
  - TorchBench focuses on performance metrics, not quality/correctness evaluation

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

- ✅ **Strategy 1: Batch Inference** - SUPPORTED
  - Primary execution mode of TorchBench
  - Runs multiple input samples through model instances
  - Supports both training and evaluation modes
  - Example: `model.invoke()` executes the model on a batch of inputs

- ❌ **Strategy 2: Arena Battle** - NOT SUPPORTED
  - No support for executing the same input across multiple models simultaneously
  - Each model runs independently

- ❌ **Strategy 3: Interactive Loop** - NOT SUPPORTED
  - No support for stateful stepping through state transitions
  - TorchBench is designed for batch execution, not interactive loops

- ❌ **Strategy 4: Production Streaming** - NOT SUPPORTED
  - No support for continuously processing live production traffic
  - TorchBench is designed for offline benchmarking only

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

- ❌ **Strategy 1: Deterministic Measurement** - NOT SUPPORTED
  - TorchBench does not provide deterministic correctness metrics (accuracy, BLEU, ROUGE, etc.)
  - Some models have `--accuracy` mode, but this is not a standardized framework feature

- ❌ **Strategy 2: Embedding Measurement** - NOT SUPPORTED
  - No support for semantic similarity calculations (BERTScore, sentence embeddings, etc.)

- ❌ **Strategy 3: Subjective Measurement** - NOT SUPPORTED
  - No support for LLM-as-judge or model-based evaluation

- ✅ **Strategy 4: Performance Measurement** - SUPPORTED
  - Core capability of TorchBench
  - Measures latency (time per batch in milliseconds)
  - Measures throughput (samples per second)
  - Measures CPU peak memory usage
  - Measures GPU peak memory usage
  - Measures FLOPs (floating point operations)
  - Measures time-to-first-batch (TTFB)
  - Measures PyTorch 2 compilation time
  - Measures PyTorch 2 graph breaks

### Step B: Aggregate Scoring

- ✅ **Strategy 1: Distributional Statistics** - SUPPORTED
  - Collects per-instance latencies and computes statistics
  - Uses pytest-benchmark for statistical analysis (mean, median, std dev)
  - Stores benchmark results in JSON format with statistical summaries

- ❌ **Strategy 2: Uncertainty Quantification** - NOT SUPPORTED
  - No support for confidence bounds, bootstrap resampling, or PPI
  - Only basic statistics are provided

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

- ✅ **Strategy 1: Execution Tracing** - SUPPORTED
  - Supports profiling with `python run.py <model> --profile`
  - Captures detailed execution information via PyTorch profiler
  - Can record shapes, memory, stack traces, FLOPs, and modules

- ❌ **Strategy 2: Subgroup Analysis** - NOT SUPPORTED
  - No built-in support for breaking down performance by subgroups
  - Results are aggregated at the model level only

- ✅ **Strategy 3: Regression Alerting** - SUPPORTED
  - Includes `regression_detector.py` for comparing results against baselines
  - Detects performance degradation and generates alerts
  - Can trigger GitHub issues for regressions

- ❌ **Strategy 4: Chart Generation** - NOT SUPPORTED
  - No built-in visualization capabilities
  - Results are stored in JSON format, but charts must be generated externally

- ❌ **Strategy 5: Dashboard Creation** - NOT SUPPORTED
  - No built-in interactive web dashboard
  - Results are stored in JSON and viewed externally (e.g., Meta's internal Unidash)

- ❌ **Strategy 6: Leaderboard Publication** - NOT SUPPORTED
  - No native support for submitting results to leaderboards
  - Results can be viewed internally at Meta, but no public leaderboard submission mechanism

---

## Supported Strategies Summary

Out of 39 strategies across all phases:

### Supported (8 strategies):
1. **Phase 0-A-1**: Git Clone installation
2. **Phase 0-A-2**: PyPI package installation
3. **Phase I-A-2**: Model-in-Process (local inference)
4. **Phase I-B-1**: Benchmark Dataset Preparation
5. **Phase II-A-1**: Batch Inference
6. **Phase III-A-4**: Performance Measurement
7. **Phase III-B-1**: Distributional Statistics
8. **Phase IV-A-1**: Execution Tracing
9. **Phase IV-A-3**: Regression Alerting

### Not Supported (31 strategies):
All other strategies listed in the unified evaluation workflow are not natively supported by TorchBench in its full installation.

---

## Conclusion

TorchBench is a specialized framework focused on **performance benchmarking of PyTorch models**. It excels at:
- Installing and loading PyTorch models locally
- Running batch inference and training
- Measuring performance metrics (latency, throughput, memory, FLOPs)
- Detecting performance regressions

TorchBench is **not designed for**:
- Remote model inference via APIs
- Correctness/accuracy evaluation
- Interactive environments or RL agents
- Multi-model comparison (arena battles)
- Production traffic monitoring
- Visualization and dashboards
- Leaderboard submissions

For evaluation workflows requiring quality assessment, semantic similarity, LLM-as-judge, or multi-modal evaluation, TorchBench would need to be extended with custom modules and external libraries.

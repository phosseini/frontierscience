# FrontierScience Benchmark - Setup & Usage Guide

## Overview

FrontierScience is a benchmark for evaluating AI on expert-level scientific reasoning, introduced by OpenAI in their blog post [*Evaluating AI's ability to perform scientific research tasks*](https://openai.com/index/frontierscience/).

This repository provides an implementation for running the FrontierScience benchmark using the dataset released by OpenAI on their [Hugging Face account](https://huggingface.co/datasets/openai/frontierscience).

- **Olympiad Track**: 100 international olympiad-level problems (short answer format)
- **Research Track**: 60 PhD-level research sub-problems (open-ended, rubric-graded)

### Paper Specifications
- **Olympiad**: 20 trials per problem, majority vote
- **Research**: 30 trials per problem, average rubric score ≥7/10 = success
- **Judge Model**: GPT-4o (for grading responses)

## Installation

```bash
# Navigate to project directory
cd frontierscience-benchmark

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## Configuration

Create a `.env` file with your API key:

```bash
cp .env.example .env
```

Edit `.env` and add:
```
OPENAI_API_KEY=sk-your-key-here
```

## Quick Test

Run a quick test (2 problems, 2 trials each):

```bash
python run_evaluation.py \
    --track research \
    --model gpt-4o \
    --limit 2 \
    --num_trials 2
```

Expected output:
```
RESEARCH RESULTS
============================================================
Model: gpt-4o
Problems evaluated: 2
Trials per problem: 2
Success threshold: 7.0/10 points
Accuracy: XX.XX%
Average rubric score: X.XX/10

Results saved to: results/
```

## Running Full Evaluations

### Research Track (as in paper)

```bash
python run_evaluation.py \
    --track research \
    --model gpt-4o \
    --judge_model gpt-4o \
    --num_trials 30 \
    --output_dir results/research
```

**Time**: 2-3 hours for 60 problems × 30 trials

### Olympiad Track (as in paper)

```bash
python run_evaluation.py \
    --track olympiad \
    --model gpt-4o \
    --num_trials 20 \
    --output_dir results/olympiad
```

**Time**: 1-2 hours for 100 problems × 20 trials

### Full Paper Reproduction

Run complete evaluation as described in the paper:

```bash
bash scripts/reproduce_paper.sh gpt-4o
```

This runs both Olympiad (100 problems × 20 trials) and Research (60 problems × 30 trials) tracks.

**Total time**: 3-5 hours

## Using Different Models

### OpenAI O1 (with reasoning effort)
```bash
python run_evaluation.py \
    --track research \
    --model o1 \
    --reasoning_effort high \
    --judge_model gpt-4o \
    --num_trials 30
```

### Claude 3.5 Sonnet
```bash
python run_evaluation.py \
    --track research \
    --model claude-3-5-sonnet-20241022 \
    --judge_model gpt-4o \
    --num_trials 30
```

### Gemini (requires GOOGLE_API_KEY in .env)
```bash
python run_evaluation.py \
    --track olympiad \
    --model gemini/gemini-2.0-flash-exp \
    --num_trials 20
```

## Analyzing Results

### Generate Summary and Plots
```bash
python analyze_results.py \
    --results_dir results/ \
    --plot \
    --output comparison.png
```

### Compare Multiple Models
```bash
python analyze_results.py \
    --results_dir results/ \
    --models "gpt-4o,claude-3-5-sonnet,o1" \
    --plot
```

### View Results
Results are saved as JSON files in the output directory. Each file contains:
- Overall accuracy/scores
- Per-problem results
- Individual trial details
- Token usage statistics

## Common Options

### Filter by Subject
```bash
python run_evaluation.py \
    --track olympiad \
    --model gpt-4o \
    --subject physics \
    --num_trials 20
```

Available subjects: `physics`, `chemistry`, `biology`

### Limit Problems (for testing)
```bash
python run_evaluation.py \
    --track research \
    --model gpt-4o \
    --limit 10 \
    --num_trials 5
```

### Custom Output Directory
```bash
python run_evaluation.py \
    --track olympiad \
    --model gpt-4o \
    --output_dir results/my-experiment
```

## Expected Results (from paper)

### GPT-4o
- Olympiad: ~61.7% accuracy
- Research: ~14.1% accuracy, ~4.2/10 avg score

### O1
- Olympiad: ~69.7% accuracy
- Research: ~20.2% accuracy, ~5.5/10 avg score

### GPT-5 (paper's best model)
- Olympiad: ~77% accuracy
- Research: ~25% accuracy, ~6.5/10 avg score

## Troubleshooting

### Rate Limit Errors
**Solution**: Wait between runs, reduce `--num_trials`, or use higher tier API access

### Module Not Found
**Solution**: Run `pip install -r requirements.txt` from project root

### API Key Not Found
**Solution**: Ensure `.env` file exists with `OPENAI_API_KEY=your-key`

### Import Errors
**Solution**: Always run commands from project root (where `src/` folder is located)

## Project Structure

```
frontierscience-benchmark/
├── data/
│   └── frontierscience_full.csv      # 160 problems
├── prompts/
│   ├── olympiad_judge_prompt.txt     # Judge prompt for olympiad
│   └── research_judge_prompt.txt     # Judge prompt for research
├── src/
│   ├── data_loader.py                # Dataset loading
│   ├── model_caller.py               # LiteLLM integration
│   └── evaluator.py                  # Evaluation logic
├── scripts/
│   └── reproduce_paper.sh            # Full paper reproduction
├── run_evaluation.py                 # Main CLI script
├── analyze_results.py                # Result analysis
├── example_usage.py                  # Code examples
└── test_installation.py              # Verify setup
```

## Programmatic Usage

```python
from src import FrontierScienceDataset, FrontierScienceEvaluator

# Load dataset
dataset = FrontierScienceDataset("data/frontierscience_full.csv")

# Create evaluator
evaluator = FrontierScienceEvaluator(
    dataset=dataset,
    model='gpt-4o',
    judge_model='gpt-4o',
    output_dir='results'
)

# Run evaluation
results = evaluator.evaluate_research(limit=10, num_trials=5)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Avg Score: {results['avg_rubric_score']:.2f}/10")
```

## Additional Resources

- **Paper**: [FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks](https://arxiv.org/abs/2501.xxxxx)
- **Dataset**: [huggingface.co/datasets/openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience)
- **LiteLLM Docs**: [docs.litellm.ai](https://docs.litellm.ai)
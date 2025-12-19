# FrontierScience Benchmark

FrontierScience is a benchmark for evaluating AI on expert-level scientific reasoning, introduced by OpenAI in their blog post [*Evaluating AI's ability to perform scientific research tasks*](https://openai.com/index/frontierscience/).

This repository provides an implementation for running the FrontierScience benchmark using the dataset released by OpenAI on their [Hugging Face account](https://huggingface.co/datasets/openai/frontierscience).

The dataset consists of 160 expert-level scientific problems organized into two difficulty tracks (Olympiad and Research) and spanning three subjects (Physics, Chemistry, and Biology):

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

## Analyzing Results

### Generate Summary and Plots
```bash
python analyze_results.py \
    --results_dir results/ \
    --plot \
    --output comparison.png
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
├── run_evaluation.py                 # Main CLI script
├── analyze_results.py                # Result analysis
└── test_installation.py              # Verify setup
```
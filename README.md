# FrontierScience Benchmark

FrontierScience is a benchmark for evaluating AI on expert-level scientific reasoning, introduced by **OpenAI** in their blog post [*Evaluating AI's ability to perform scientific research tasks*](https://openai.com/index/frontierscience/).

This repository provides an implementation for running the FrontierScience benchmark using the dataset released by OpenAI on their [Hugging Face](https://huggingface.co/datasets/openai/frontierscience). The dataset consists of 160 expert-level scientific problems organized into two tracks (Olympiad and Research) and spanning three subjects (Physics, Chemistry, and Biology):

- **Olympiad Track**: 100 international olympiad-level problems (short answer format)
- **Research Track**: 60 PhD-level research sub-problems (open-ended, rubric-graded)

### Evaluation Specifications
- **Olympiad**: 20 trials per problem, majority vote
- **Research**: 30 trials per problem, average rubric score ≥7/10 = success
- **Judge Model**: GPT-5 thinking at high reasoning effort

## Installation

```bash
# Navigate to project directory
cd frontierscience

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

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional, for using Claude models
```

## Quick Test

Run a quick test (2 problems, 2 trials each):

```bash
python run_evaluation.py \
    --track research \
    --model gpt-5-mini-2025-08-07 \
    --limit 2 \
    --num_trials 2
```

## Command-Line Arguments

The `run_evaluation.py` script accepts the following arguments:

### Required Arguments
- `--track`: Evaluation track to run
  - Choices: `olympiad`, `research`
  - Example: `--track research`

- `--model`: Model to evaluate
  - Examples: `gpt-5.2-2025-12-11`, `claude-sonnet-4-5-20250929`
  - Example: `--model gpt-4o`

### Optional Arguments
- `--data_path`: Path to dataset CSV file
  - Default: `data/frontierscience.csv`
  - Example: `--data_path data/frontierscience.csv`

- `--judge_model`: Model to use as judge for grading
  - Default: `gpt-5-2025-08-07`
  - Example: `--judge_model gpt-5.2-2025-12-11`

- `--reasoning_effort`: Reasoning effort level for reasoning models (gpt-5*, o1, o3)
  - Choices: `low`, `medium`, `high`
  - Example: `--reasoning_effort high`

- `--subject`: Filter problems by subject
  - Choices: `physics`, `chemistry`, `biology`
  - Example: `--subject physics`

- `--limit`: Limit number of problems to evaluate (useful for testing)
  - Example: `--limit 5`

- `--num_trials`: Number of trials per problem
  - Default: 20 for olympiad track, 30 for research track
  - Example: `--num_trials 10`

- `--output_dir`: Directory to save results
  - Default: `results/`
  - Example: `--output_dir results/my-experiment`

- `--success_threshold`: Rubric score threshold for research track success
  - Default: `7.0` (out of 10 points)
  - Example: `--success_threshold 8.0`

- `--verbose`: Print judge model outputs for debugging
  - Flag (no value needed)
  - Example: `--verbose`

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

## Project Structure

```
frontierscience-benchmark/
├── data/
│   └── frontierscience.csv           # 160 problems
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
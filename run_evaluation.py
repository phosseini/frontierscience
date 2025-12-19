#!/usr/bin/env python3
"""
Main evaluation script for FrontierScience benchmark.

Usage examples:
  # Olympiad track with gpt-4o
  python run_evaluation.py --track olympiad --model gpt-4o
  
  # Research track with o1 at high reasoning effort
  python run_evaluation.py --track research --model o1 --reasoning_effort high --judge_model gpt-4o
  
  # Filter by subject
  python run_evaluation.py --track olympiad --model gpt-4o --subject physics
  
  # Limit number of problems for testing
  python run_evaluation.py --track research --model gpt-4o --limit 5 --num_trials 3
"""

import argparse
from pathlib import Path

from src import FrontierScienceDataset, FrontierScienceEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLMs on FrontierScience benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--track',
        type=str,
        required=True,
        choices=['olympiad', 'research'],
        help='Evaluation track to run'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model to evaluate (e.g., gpt-4o, claude-3-5-sonnet-20241022, o1)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/frontierscience.csv',
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--judge_model',
        type=str,
        default='gpt-5-2025-08-07',
        help='Model to use as judge for grading (default: gpt-5-2025-08-07)'
    )
    
    parser.add_argument(
        '--reasoning_effort',
        type=str,
        choices=['low', 'medium', 'high'],
        help='Reasoning effort (low, medium, high)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        help='Filter problems by subject (e.g., physics, chemistry, biology)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of problems to evaluate (for testing)'
    )
    
    parser.add_argument(
        '--num_trials',
        type=int,
        help='Number of trials per problem (default: 20 for olympiad, 30 for research)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results (default: results/)'
    )
    
    parser.add_argument(
        '--success_threshold',
        type=float,
        default=7.0,
        help='Rubric score threshold for research track success (default: 7.0)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print judge model outputs for debugging'
    )

    args = parser.parse_args()
    
    # Validate data path
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found at {args.data_path}")
        print("Please ensure frontierscience.csv is in the data/ directory")
        return 1
    
    # Set default num_trials based on track
    if args.num_trials is None:
        args.num_trials = 20 if args.track == 'olympiad' else 30
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = FrontierScienceDataset(args.data_path)
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  Olympiad: {stats['olympiad_problems']}")
    print(f"  Research: {stats['research_problems']}")
    print(f"  Subjects: {list(stats['subjects'].keys())}")
    
    # Initialize evaluator
    print(f"\nInitializing evaluator...")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Track: {args.track}")
    if args.reasoning_effort:
        print(f"  Reasoning effort: {args.reasoning_effort}")
    if args.verbose:
        print(f"  Verbose mode: ENABLED (will print judge outputs)")
    
    evaluator = FrontierScienceEvaluator(
        dataset=dataset,
        model=args.model,
        judge_model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting {args.track.upper()} evaluation")
    print(f"{'='*60}\n")
    
    try:
        if args.track == 'olympiad':
            results = evaluator.evaluate_olympiad(
                subject=args.subject,
                limit=args.limit,
                num_trials=args.num_trials
            )
            
            print(f"\n{'='*60}")
            print("OLYMPIAD RESULTS")
            print(f"{'='*60}")
            print(f"Model: {results['model']}")
            print(f"Problems evaluated: {results['num_problems']}")
            print(f"Trials per problem: {results['num_trials']}")
            print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
            
        else:  # research
            results = evaluator.evaluate_research(
                subject=args.subject,
                limit=args.limit,
                num_trials=args.num_trials,
                success_threshold=args.success_threshold
            )
            
            print(f"\n{'='*60}")
            print("RESEARCH RESULTS")
            print(f"{'='*60}")
            print(f"Model: {results['model']}")
            print(f"Problems evaluated: {results['num_problems']}")
            print(f"Trials per problem: {results['num_trials']}")
            print(f"Success threshold: {results['success_threshold']}/10 points")
            print(f"Accuracy: {results['accuracy']:.2%} ({results['successful']}/{results['total']})")
            print(f"Average rubric score: {results['avg_rubric_score']:.2f}/10")
        
        print(f"\nResults saved to: {args.output_dir}/")
        return 0
    
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
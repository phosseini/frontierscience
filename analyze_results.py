#!/usr/bin/env python3
"""
Analyze and visualize FrontierScience evaluation results.

Usage:
  # Analyze all results in a directory
  python analyze_results.py --results_dir results/
  
  # Compare specific models
  python analyze_results.py --results_dir results/ --models gpt-4o,claude-3-5-sonnet,o1
  
  # Generate comparison plots
  python analyze_results.py --results_dir results/ --plot --output comparison.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: Path) -> List[Dict]:
    """Load all result JSON files from directory."""
    results = []
    
    for json_file in results_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                result['filename'] = json_file.name
                results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results


def summarize_results(results: List[Dict]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    summaries = []
    
    for result in results:
        summary = {
            'model': result['model'],
            'track': result['track'],
            'num_problems': result.get('num_problems', 0),
            'num_trials': result.get('num_trials', 0),
            'timestamp': result.get('timestamp', 'N/A')
        }
        
        if result['track'] == 'olympiad':
            summary['accuracy'] = result.get('accuracy', 0)
            summary['correct'] = result.get('correct', 0)
            summary['total'] = result.get('total', 0)
        else:  # research
            summary['accuracy'] = result.get('accuracy', 0)
            summary['avg_rubric_score'] = result.get('avg_rubric_score', 0)
            summary['success_threshold'] = result.get('success_threshold', 7.0)
            summary['successful'] = result.get('successful', 0)
            summary['total'] = result.get('total', 0)
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def analyze_by_subject(results: List[Dict]) -> pd.DataFrame:
    """Analyze results broken down by subject."""
    subject_data = []
    
    for result in results:
        model = result['model']
        track = result['track']
        
        for problem_result in result.get('results', []):
            subject = problem_result.get('subject', 'unknown')
            
            if track == 'olympiad':
                subject_data.append({
                    'model': model,
                    'track': track,
                    'subject': subject,
                    'correct': problem_result.get('correct', False)
                })
            else:  # research
                subject_data.append({
                    'model': model,
                    'track': track,
                    'subject': subject,
                    'rubric_score': problem_result.get('avg_rubric_score', 0),
                    'success': problem_result.get('success', False)
                })
    
    return pd.DataFrame(subject_data)


def plot_model_comparison(df: pd.DataFrame, output_path: str = None):
    """Create comparison plots for models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Olympiad accuracy
    olympiad_df = df[df['track'] == 'olympiad'].copy()
    if not olympiad_df.empty:
        olympiad_df = olympiad_df.sort_values('accuracy', ascending=False)
        axes[0].barh(olympiad_df['model'], olympiad_df['accuracy'] * 100)
        axes[0].set_xlabel('Accuracy (%)')
        axes[0].set_title('FrontierScience-Olympiad Performance')
        axes[0].set_xlim(0, 100)
        
        for i, (idx, row) in enumerate(olympiad_df.iterrows()):
            axes[0].text(row['accuracy'] * 100 + 1, i, f"{row['accuracy']*100:.1f}%", 
                        va='center')
    
    # Plot 2: Research track
    research_df = df[df['track'] == 'research'].copy()
    if not research_df.empty:
        research_df = research_df.sort_values('accuracy', ascending=False)
        
        x = range(len(research_df))
        width = 0.35
        
        axes[1].barh([i - width/2 for i in x], research_df['accuracy'] * 100, 
                    width, label='Success Rate (%)', alpha=0.8)
        axes[1].barh([i + width/2 for i in x], research_df['avg_rubric_score'] * 10, 
                    width, label='Avg Score (scaled to %)', alpha=0.8)
        
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(research_df['model'])
        axes[1].set_xlabel('Percentage / Score')
        axes[1].set_title('FrontierScience-Research Performance')
        axes[1].legend()
        axes[1].set_xlim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def plot_subject_breakdown(subject_df: pd.DataFrame, output_path: str = None):
    """Create subject-wise breakdown plots."""
    olympiad_df = subject_df[subject_df['track'] == 'olympiad'].copy()
    research_df = subject_df[subject_df['track'] == 'research'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Olympiad by subject
    if not olympiad_df.empty:
        olympiad_pivot = olympiad_df.groupby(['model', 'subject'])['correct'].mean().unstack()
        olympiad_pivot.plot(kind='bar', ax=axes[0], alpha=0.8)
        axes[0].set_title('Olympiad Accuracy by Subject')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Model')
        axes[0].legend(title='Subject')
        axes[0].set_ylim(0, 1)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Research by subject
    if not research_df.empty:
        research_pivot = research_df.groupby(['model', 'subject'])['rubric_score'].mean().unstack()
        research_pivot.plot(kind='bar', ax=axes[1], alpha=0.8)
        axes[1].set_title('Research Avg Rubric Score by Subject')
        axes[1].set_ylabel('Average Score (out of 10)')
        axes[1].set_xlabel('Model')
        axes[1].legend(title='Subject')
        axes[1].set_ylim(0, 10)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Subject breakdown plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FrontierScience evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing result JSON files'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of models to filter (optional)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='comparison.png',
        help='Output filename for plots (default: comparison.png)'
    )
    
    parser.add_argument(
        '--subject_plot',
        type=str,
        help='Output filename for subject breakdown plot'
    )
    
    args = parser.parse_args()
    
    # Load results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return 1
    
    print(f"Loaded {len(results)} result files\n")
    
    # Filter by models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]
        results = [r for r in results if r['model'] in model_list]
        print(f"Filtered to {len(results)} results for models: {model_list}\n")
    
    # Create summary
    summary_df = summarize_results(results)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()
    
    # Analyze by subject
    subject_df = analyze_by_subject(results)
    
    if not subject_df.empty:
        print("=" * 80)
        print("BREAKDOWN BY SUBJECT")
        print("=" * 80)

        # Olympiad
        olympiad_df = subject_df[subject_df['track'] == 'olympiad']
        if not olympiad_df.empty:
            olympiad_subject = olympiad_df.groupby(
                ['model', 'subject']
            )['correct'].mean().unstack()

            print("\nOlympiad Track:")
            print(olympiad_subject.to_string())

        # Research
        research_df = subject_df[subject_df['track'] == 'research']
        if not research_df.empty:
            research_subject = research_df.groupby(
                ['model', 'subject']
            )['rubric_score'].mean().unstack()

            print("\nResearch Track (Avg Rubric Score):")
            print(research_subject.to_string())
        print()
    
    # Generate plots
    if args.plot:
        print("Generating comparison plots...")
        plot_model_comparison(summary_df, args.output)
        
        if args.subject_plot and not subject_df.empty:
            plot_subject_breakdown(subject_df, args.subject_plot)
    
    return 0


if __name__ == "__main__":
    exit(main())
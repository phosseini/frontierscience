"""
Evaluator for FrontierScience benchmark.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from src.data_loader import FrontierScienceDataset
from src.model_caller import ModelCaller


class FrontierScienceEvaluator:
    """Evaluate models on FrontierScience benchmark."""
    
    def __init__(
        self,
        dataset: FrontierScienceDataset,
        model: str,
        judge_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        output_dir: str = "results",
        verbose: bool = False
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: FrontierScienceDataset instance
            model: Model to evaluate
            judge_model: Model to use as judge (for grading responses)
            reasoning_effort: For reasoning models: 'low', 'medium', 'high'
            output_dir: Directory to save results
            verbose: If True, print judge model outputs for debugging
        """
        self.dataset = dataset
        self.model = model
        self.judge_model = judge_model or 'gpt-4o'
        self.reasoning_effort = reasoning_effort
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Initialize model callers
        self.model_caller = ModelCaller(
            model=model,
            reasoning_effort=reasoning_effort
        )

        self.judge_caller = ModelCaller(
            model=self.judge_model,
            reasoning_effort='high' if 'o1' in self.judge_model or 'o3' in self.judge_model else None
        )
        
        # Load prompts
        self.olympiad_judge_prompt = self._load_prompt('prompts/olympiad_judge_prompt.txt')
        self.research_judge_prompt = self._load_prompt('prompts/research_judge_prompt.txt')
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load a prompt template from file."""
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return path.read_text()
    
    def evaluate_olympiad(
        self,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
        num_trials: int = 20
    ) -> Dict:
        """
        Evaluate on Olympiad track.
        
        Args:
            subject: Filter by subject (optional)
            limit: Limit number of problems (optional)
            num_trials: Number of independent trials per problem (paper uses 20)
        
        Returns:
            Dictionary with evaluation results
        """
        problems = self.dataset.get_olympiad_problems(subject=subject)
        if limit:
            problems = problems[:limit]
        
        print(f"Evaluating {len(problems)} Olympiad problems with {num_trials} trials each...")

        results = []
        problems_bar = tqdm(problems, desc="Problems", position=0, disable=self.verbose)
        for problem_idx, problem_data in enumerate(problems_bar):
            problems_bar.set_description(f"Problem {problem_idx+1}/{len(problems)}")
            problem_results = self._evaluate_problem_olympiad(
                problem_data,
                num_trials=num_trials
            )
            results.append(problem_results)
        
        # Compute aggregate statistics
        accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
        
        eval_results = {
            'track': 'olympiad',
            'model': self.model,
            'judge_model': self.judge_model,
            'num_problems': len(results),
            'num_trials': num_trials,
            'accuracy': accuracy,
            'correct': sum(r['correct'] for r in results),
            'total': len(results),
            'subject_filter': subject,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        self._save_results(eval_results, 'olympiad')
        
        return eval_results
    
    def _evaluate_problem_olympiad(
        self,
        problem_data: Dict,
        num_trials: int = 20
    ) -> Dict:
        """Evaluate a single Olympiad problem with multiple trials."""
        problem = problem_data['problem']
        reference_answer = problem_data['answer']

        trial_results = []

        trials_bar = tqdm(range(num_trials), desc="  Trials", position=1, leave=False, disable=self.verbose)
        for trial in trials_bar:
            try:
                # Get model's answer
                response = self.model_caller.call(prompt=problem)

                attempted_answer = response['content']
                
                # Judge the answer
                judge_result = self._judge_olympiad_answer(
                    problem=problem,
                    reference_answer=reference_answer,
                    attempted_answer=attempted_answer
                )
                
                trial_results.append({
                    'trial': trial,
                    'attempted_answer': attempted_answer,
                    'correct': judge_result['correct'],
                    'judge_reasoning': judge_result['reasoning'],
                    'usage': response['usage']
                })
            
            except Exception as e:
                print(f"Error in trial {trial}: {str(e)}")
                trial_results.append({
                    'trial': trial,
                    'error': str(e),
                    'correct': False
                })
        
        # Determine if problem was solved (majority vote across trials)
        correct_count = sum(t['correct'] for t in trial_results if 'correct' in t)
        
        return {
            'problem': problem[:200] + '...',  # Truncate for storage
            'reference_answer': reference_answer,
            'subject': problem_data.get('subject'),
            'task_group_id': problem_data.get('task_group_id'),
            'correct': correct_count > (num_trials / 2),  # Majority vote
            'correct_trials': correct_count,
            'total_trials': num_trials,
            'trials': trial_results
        }
    
    def _judge_olympiad_answer(
        self,
        problem: str,
        reference_answer: str,
        attempted_answer: str
    ) -> Dict:
        """Judge an Olympiad answer using the judge model."""
        judge_prompt = self.olympiad_judge_prompt.format(
            problem=problem,
            reference_answer=reference_answer,
            answer=attempted_answer
        )

        response = self.judge_caller.call(prompt=judge_prompt)

        content = response['content']

        if self.verbose:
            print("\n" + "="*80, flush=True)
            print("OLYMPIAD JUDGE OUTPUT:", flush=True)
            print("="*80, flush=True)
            print(content, flush=True)
            print("="*80 + "\n", flush=True)

        # Parse verdict
        correct = 'VERDICT: CORRECT' in content

        return {
            'correct': correct,
            'reasoning': content
        }
    
    def evaluate_research(
        self,
        subject: Optional[str] = None,
        limit: Optional[int] = None,
        num_trials: int = 30,
        success_threshold: float = 7.0
    ) -> Dict:
        """
        Evaluate on Research track.
        
        Args:
            subject: Filter by subject (optional)
            limit: Limit number of problems (optional)
            num_trials: Number of independent trials per problem (paper uses 30)
            success_threshold: Rubric points threshold for success (paper uses 7/10)
        
        Returns:
            Dictionary with evaluation results
        """
        problems = self.dataset.get_research_problems(subject=subject)
        if limit:
            problems = problems[:limit]
        
        print(f"Evaluating {len(problems)} Research problems with {num_trials} trials each...")

        results = []
        problems_bar = tqdm(problems, desc="Problems", position=0, disable=self.verbose)
        for problem_idx, problem_data in enumerate(problems_bar):
            problems_bar.set_description(f"Problem {problem_idx+1}/{len(problems)}")
            problem_results = self._evaluate_problem_research(
                problem_data,
                num_trials=num_trials,
                success_threshold=success_threshold
            )
            results.append(problem_results)
        
        # Compute aggregate statistics
        accuracy = sum(r['success'] for r in results) / len(results) if results else 0
        avg_rubric_score = sum(r['avg_rubric_score'] for r in results) / len(results) if results else 0
        
        eval_results = {
            'track': 'research',
            'model': self.model,
            'judge_model': self.judge_model,
            'num_problems': len(results),
            'num_trials': num_trials,
            'success_threshold': success_threshold,
            'accuracy': accuracy,
            'avg_rubric_score': avg_rubric_score,
            'successful': sum(r['success'] for r in results),
            'total': len(results),
            'subject_filter': subject,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        self._save_results(eval_results, 'research')
        
        return eval_results
    
    def _evaluate_problem_research(
        self,
        problem_data: Dict,
        num_trials: int = 30,
        success_threshold: float = 7.0
    ) -> Dict:
        """Evaluate a single Research problem with multiple trials."""
        problem = problem_data['problem']
        rubric = problem_data['answer']  # For research track, 'answer' contains the rubric

        trial_results = []

        trials_bar = tqdm(range(num_trials), desc="  Trials", position=1, leave=False, disable=self.verbose)
        for trial in trials_bar:
            try:
                # Get model's answer
                response = self.model_caller.call(prompt=problem)

                attempted_answer = response['content']

                # Judge the answer using rubric
                judge_result = self._judge_research_answer(
                    problem=problem,
                    rubric=rubric,
                    attempted_answer=attempted_answer
                )
                
                trial_results.append({
                    'trial': trial,
                    'attempted_answer': attempted_answer,
                    'rubric_score': judge_result['rubric_score'],
                    'judge_reasoning': judge_result['reasoning'],
                    'usage': response['usage']
                })
            
            except Exception as e:
                print(f"Error in trial {trial}: {str(e)}")
                trial_results.append({
                    'trial': trial,
                    'error': str(e),
                    'rubric_score': 0
                })
        
        # Calculate average rubric score across trials
        avg_score = sum(t.get('rubric_score', 0) for t in trial_results) / len(trial_results)
        
        return {
            'problem': problem[:200] + '...',  # Truncate for storage
            'rubric': rubric[:200] + '...',
            'subject': problem_data.get('subject'),
            'task_group_id': problem_data.get('task_group_id'),
            'avg_rubric_score': avg_score,
            'success': avg_score >= success_threshold,
            'num_trials': num_trials,
            'trials': trial_results
        }
    
    def _judge_research_answer(
        self,
        problem: str,
        rubric: str,
        attempted_answer: str
    ) -> Dict:
        """Judge a Research answer using rubric and judge model."""
        judge_prompt = self.research_judge_prompt.format(
            problem=problem,
            rubric=rubric,
            answer=attempted_answer
        )

        response = self.judge_caller.call(prompt=judge_prompt)

        content = response['content']

        if self.verbose:
            print("\n" + "="*80, flush=True)
            print("RESEARCH JUDGE OUTPUT:", flush=True)
            print("="*80, flush=True)
            print(content, flush=True)
            print("="*80 + "\n", flush=True)

        # Parse rubric score from VERDICT: X.X
        # Note: -1 indicates a parsing error (to distinguish from legitimate 0 scores)
        rubric_score = -1.0
        if 'VERDICT:' in content:
            try:
                verdict_line = [line for line in content.split('\n') if 'VERDICT:' in line][0]
                score_str = verdict_line.split('VERDICT:')[1].strip()
                # Remove any markdown formatting (**, *, etc.) and extract just the number
                import re
                score_match = re.search(r'[\d.]+', score_str)
                if score_match:
                    rubric_score = float(score_match.group())
                    # Validate score is within range [0, 10]
                    if rubric_score < 0 or rubric_score > 10:
                        print(f"Warning: Rubric score {rubric_score} is outside valid range [0, 10]. Setting to -1 (error).")
                        rubric_score = -1.0
                else:
                    raise ValueError(f"No numeric score found in: {score_str}")
            except (IndexError, ValueError) as e:
                print(f"Error parsing rubric score: {e}")
                if self.verbose:
                    print(f"Problematic verdict line: {verdict_line if 'verdict_line' in locals() else 'N/A'}")
                rubric_score = -1.0
        else:
            print(f"Warning: No VERDICT found in judge output. Setting score to -1 (error).")

        # Final validation: ensure rubric_score is a number
        if not isinstance(rubric_score, (int, float)):
            print(f"Warning: rubric_score is not a number (type: {type(rubric_score)}). Setting to -1 (error).")
            rubric_score = -1.0

        return {
            'rubric_score': rubric_score,
            'reasoning': content
        }
    
    def _save_results(self, results: Dict, track: str):
        """Save evaluation results to JSON file."""
        filename = f"{track}_{self.model.replace('/', '_')}_{results['timestamp'].replace(' ', '_').replace(':', '-')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data_loader import FrontierScienceDataset
    
    dataset = FrontierScienceDataset("data/frontierscience_full.csv")
    
    evaluator = FrontierScienceEvaluator(
        dataset=dataset,
        model='gpt-4o',
        judge_model='gpt-4o',
        output_dir='results/test'
    )
    
    # Run on a subset for testing
    results = evaluator.evaluate_research(limit=1, num_trials=2)
    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"Avg Rubric Score: {results['avg_rubric_score']:.2f}/10")
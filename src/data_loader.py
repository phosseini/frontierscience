"""
Data loader for FrontierScience benchmark dataset.
"""
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class FrontierScienceDataset:
    """Load and manage FrontierScience benchmark data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the frontierscience_full.csv file
        """
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        self._validate_data()
    
    def _validate_data(self):
        """Validate the dataset structure."""
        required_columns = ['problem', 'answer', 'subject', 'task_group_id', 'category']
        missing_columns = set(required_columns) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        valid_categories = ['olympiad', 'research']
        invalid_categories = set(self.df['category'].unique()) - set(valid_categories)
        if invalid_categories:
            raise ValueError(f"Invalid categories found: {invalid_categories}")
    
    def get_problems(
        self, 
        category: Optional[str] = None,
        subject: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get problems filtered by category and/or subject.
        
        Args:
            category: Filter by 'olympiad' or 'research' (None = all)
            subject: Filter by subject (e.g., 'physics', 'chemistry', 'biology')
            limit: Maximum number of problems to return
        
        Returns:
            List of problem dictionaries
        """
        df_filtered = self.df.copy()
        
        if category:
            if category not in ['olympiad', 'research']:
                raise ValueError(f"Invalid category: {category}")
            df_filtered = df_filtered[df_filtered['category'] == category]
        
        if subject:
            df_filtered = df_filtered[df_filtered['subject'] == subject]
        
        if limit:
            df_filtered = df_filtered.head(limit)
        
        return df_filtered.to_dict('records')
    
    def get_olympiad_problems(self, subject: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get all Olympiad track problems."""
        return self.get_problems(category='olympiad', subject=subject, limit=limit)
    
    def get_research_problems(self, subject: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get all Research track problems."""
        return self.get_problems(category='research', subject=subject, limit=limit)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_problems': len(self.df),
            'olympiad_problems': len(self.df[self.df['category'] == 'olympiad']),
            'research_problems': len(self.df[self.df['category'] == 'research']),
            'subjects': self.df['subject'].value_counts().to_dict(),
            'category_by_subject': self.df.groupby(['category', 'subject']).size().to_dict()
        }
        return stats


if __name__ == "__main__":
    # Example usage
    dataset = FrontierScienceDataset("data/frontierscience_full.csv")
    print("Dataset Statistics:")
    print(dataset.get_statistics())
    
    print("\nFirst Research Problem:")
    research_problems = dataset.get_research_problems(limit=1)
    print(research_problems[0]['problem'][:200] + "...")
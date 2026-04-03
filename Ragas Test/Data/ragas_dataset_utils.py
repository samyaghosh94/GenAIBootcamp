"""
Utility functions for working with RAGAS evaluation datasets.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


class RAGASDatasetUtils:
    """
    Utilities for creating, loading, and managing RAGAS evaluation datasets.
    """
    
    @staticmethod
    def load_dataset(filepath: str) -> List[Dict[str, Any]]:
        """Load a RAGAS dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_dataset(dataset: List[Dict[str, Any]], filepath: str):
        """Save a RAGAS dataset to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def validate_dataset(dataset: List[Dict[str, Any]]) -> bool:
        """
        Validate that dataset has required fields for RAGAS evaluation.
        Required fields: question, ground_truth, contexts
        """
        required_fields = {'question', 'ground_truth', 'contexts'}
        
        for i, item in enumerate(dataset):
            if not isinstance(item, dict):
                print(f"❌ Item {i} is not a dictionary")
                return False
            
            if not required_fields.issubset(item.keys()):
                missing = required_fields - set(item.keys())
                print(f"❌ Item {i} missing fields: {missing}")
                return False
            
            if not isinstance(item['contexts'], list):
                print(f"❌ Item {i}: contexts must be a list")
                return False
        
        return True
    
    @staticmethod
    def get_dataset_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not dataset:
            return {}
        
        total_items = len(dataset)
        total_contexts = sum(len(item['contexts']) for item in dataset)
        avg_contexts = total_contexts / total_items if total_items > 0 else 0
        
        total_question_len = sum(len(item['question']) for item in dataset)
        avg_question_len = total_question_len / total_items if total_items > 0 else 0
        
        total_context_len = sum(
            sum(len(ctx) for ctx in item['contexts']) 
            for item in dataset
        )
        avg_context_len = total_context_len / total_contexts if total_contexts > 0 else 0
        
        return {
            'total_items': total_items,
            'total_contexts': total_contexts,
            'avg_contexts_per_item': round(avg_contexts, 2),
            'avg_question_length': round(avg_question_len, 2),
            'avg_context_length': round(avg_context_len, 2),
        }
    
    @staticmethod
    def filter_by_source(dataset: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
        """Filter dataset items by source file."""
        return [item for item in dataset if item.get('source_file') == source_file]
    
    @staticmethod
    def split_dataset(dataset: List[Dict[str, Any]], 
                     train_ratio: float = 0.8) -> tuple:
        """
        Split dataset into training and evaluation sets.
        
        Args:
            dataset: Full dataset
            train_ratio: Ratio for training set (0.0 to 1.0)
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        split_idx = int(len(dataset) * train_ratio)
        return dataset[:split_idx], dataset[split_idx:]
    
    @staticmethod
    def merge_datasets(*datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge multiple RAGAS datasets into one."""
        merged = []
        for dataset in datasets:
            merged.extend(dataset)
        return merged


if __name__ == "__main__":
    # Example usage
    print("RAGAS Dataset Utilities")
    print("=" * 50)
    
    # Example: Load and validate
    # dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")
    # if RAGASDatasetUtils.validate_dataset(dataset):
    #     print("✅ Dataset is valid")
    #     stats = RAGASDatasetUtils.get_dataset_stats(dataset)
    #     print(f"📊 Dataset Statistics: {stats}")


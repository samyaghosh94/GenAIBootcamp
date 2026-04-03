# RAGAS Evaluation Dataset Generator

This module provides tools to create RAGAS (Retrieval-Augmented Generation Assessment) evaluation datasets from DOCX files.

## Overview

RAGAS is a framework for evaluating Retrieval-Augmented Generation (RAG) systems. This toolkit helps you:

1. **Load DOCX files** from a storage folder
2. **Chunk the documents** into manageable pieces
3. **Extract or generate questions** from the content
4. **Create evaluation samples** in RAGAS format
5. **Save the dataset** for use in evaluation workflows

## Files

- **dataset.py** - Main script with RAGAS dataset generation functions
- **generate_ragas_dataset.py** - Standalone script to generate datasets
- **ragas_dataset_utils.py** - Utility functions for working with RAGAS datasets

## Installation

### Required Dependencies

```bash
pip install python-dotenv langchain-unstructured langchain-text-splitters unstructured
```

### Optional Dependencies (for advanced features)

```bash
pip install python-magic-bin pillow  # For unstructured document processing
pip install pdf2image pdfmium  # If working with PDFs
```

## Quick Start

### Method 1: Using the Main Script

```bash
cd "C:\dev\GenAIBootcamp\Ragas Test\Data"
python dataset.py
```

This will:
1. Load all DOCX files from `./storage`
2. Generate the RAGAS dataset
3. Save it to `./ragas_evaluation_dataset.json`

### Method 2: Using the Standalone Generator

```bash
python generate_ragas_dataset.py
```

### Method 3: Using in Your Code

```python
from dataset import generate_ragas_evaluation_dataset

# Generate dataset
dataset = generate_ragas_evaluation_dataset(
    storage_folder="./storage",
    output_path="./ragas_evaluation_dataset.json"
)

print(f"Generated {len(dataset)} evaluation samples")
```

## RAGAS Dataset Format

Each item in the dataset has the following structure:

```json
{
    "question": "How to add a new employee record?",
    "ground_truth": "To add a new employee record...",
    "contexts": [
        "First context chunk from document...",
        "Second relevant context chunk...",
        "Third relevant context chunk..."
    ],
    "source_file": "HCM_BackOffice_User_Help_Guide_Enhanced.docx"
}
```

### Required Fields:
- **question**: The evaluation query/question
- **ground_truth**: The expected answer or reference text
- **contexts**: List of document chunks that should be retrieved for this question

### Optional Fields:
- **source_file**: The source document where the question came from
- **answer**: System-generated answer (for evaluation)

## Using the Utilities

### Load and Validate Dataset

```python
from ragas_dataset_utils import RAGASDatasetUtils

# Load dataset
dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")

# Validate dataset
if RAGASDatasetUtils.validate_dataset(dataset):
    print("✅ Dataset is valid")
```

### Get Dataset Statistics

```python
stats = RAGASDatasetUtils.get_dataset_stats(dataset)
print(f"Dataset Statistics:")
print(f"  Total Items: {stats['total_items']}")
print(f"  Total Contexts: {stats['total_contexts']}")
print(f"  Average Contexts per Item: {stats['avg_contexts_per_item']}")
print(f"  Average Question Length: {stats['avg_question_length']}")
print(f"  Average Context Length: {stats['avg_context_length']}")
```

### Filter by Source File

```python
# Get only items from a specific file
filtered = RAGASDatasetUtils.filter_by_source(
    dataset, 
    "HCM_Portal_User_Guide.docx"
)
```

### Split Dataset

```python
# Split into 80% train, 20% eval
train_set, eval_set = RAGASDatasetUtils.split_dataset(dataset, train_ratio=0.8)

# Save separately
RAGASDatasetUtils.save_dataset(train_set, "ragas_train_dataset.json")
RAGASDatasetUtils.save_dataset(eval_set, "ragas_eval_dataset.json")
```

### Merge Multiple Datasets

```python
# Merge datasets from different sources
merged = RAGASDatasetUtils.merge_datasets(dataset1, dataset2, dataset3)
```

## Configuration

You can customize the dataset generation by modifying these parameters in `generate_ragas_dataset.py`:

```python
# Configuration
STORAGE_FOLDER = "./storage"           # Where DOCX files are located
OUTPUT_DATASET_PATH = "./ragas_evaluation_dataset.json"  # Output file
CHUNK_SIZE = 1000                      # Characters per chunk
CHUNK_OVERLAP = 200                    # Overlap between chunks
```

## Question Generation Strategy

The system generates questions using multiple strategies:

1. **Q&A Pattern Matching**: Looks for existing Q&A patterns in documents
   - "Q: ...", "Question: ...", "How ...", "What ...", etc.

2. **Heading-Based Generation**: Creates questions from document headings
   - Converts "Getting Started" → "How to getting started?"

3. **Keyword-Based Generation**: Generates questions from relevant sentences
   - Extracts sentences containing keywords like "how", "what", "user", "step", "process"

## Example Output

When running the script, you'll see output like:

```
🚀 Starting RAGAS Dataset Generation
==================================================

[Step 1] Loading DOCX files...
📄 Found 3 DOCX files
📂 Loading: HCM_BackOffice_User_Help_Guide_Enhanced.docx
✅ Loaded 45 elements from HCM_BackOffice_User_Help_Guide_Enhanced.docx
📂 Loading: HCM_Portal_User_Guide.docx
✅ Loaded 38 elements from HCM_Portal_User_Guide.docx
...
✅ Total documents loaded: 145

[Step 2] Chunking documents...
✂️ Created 234 chunks from documents

[Step 3] Creating RAGAS evaluation dataset...
📊 Processing HCM_BackOffice_User_Help_Guide_Enhanced.docx (78 chunks)
📊 Processing HCM_Portal_User_Guide.docx (79 chunks)
📊 Processing HCM-Portal-TroubleshootingGuid.docx (77 chunks)

[Step 4] Saving dataset...
✅ RAGAS dataset saved to: ./ragas_evaluation_dataset.json
📊 Dataset contains 42 evaluation samples

==================================================
✅ RAGAS dataset generation completed successfully!
```

## Integration with RAGAS Framework

Once you have your dataset, you can use it with the RAGAS evaluation framework:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Load your dataset
dataset = RAGASDatasetUtils.load_dataset("ragas_evaluation_dataset.json")

# Evaluate your RAG system
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    # Add your LLM and embedding model here
)

print(results)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain_unstructured'"
**Solution**: Install the required packages
```bash
pip install langchain-unstructured unstructured
```

### Issue: "No documents loaded"
**Solution**: Ensure DOCX files are in the storage folder with .docx extension
```bash
# Check storage folder
dir ./storage
```

### Issue: "No dataset items created"
**Solution**: This means no questions were extracted. Try:
1. Verify documents have readable content
2. Increase max_questions parameter in `extract_questions_from_content()`
3. Check question extraction patterns in the function

## Performance Tips

- **Large Documents**: For very large DOCX files, reduce `CHUNK_SIZE` to 500-750
- **Many Documents**: Process in batches and merge datasets using `RAGASDatasetUtils.merge_datasets()`
- **Memory Issues**: Increase `CHUNK_OVERLAP` to reduce total chunks if memory is limited

## Support

For issues or questions about RAGAS:
- Check the [RAGAS GitHub repository](https://github.com/explodinggradients/ragas)
- Read the [RAGAS documentation](https://docs.ragas.io/)

For LangChain-related issues:
- Check [LangChain documentation](https://python.langchain.com/)


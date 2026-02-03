# RAGAS Evaluation for RAG Pipeline

This directory contains scripts to evaluate your RAG pipeline using the RAGAS (Retrieval Augmented Generation Assessment) library.

## 📋 Overview

The evaluation process has two main steps:

1. **Run RAG Pipeline** (`run_ragas_evaluation.py`) - Executes your RAG on test questions and collects answers + contexts
2. **Evaluate with RAGAS** (`evaluate_with_ragas.py`) - Computes metrics to assess RAG performance

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install ragas langchain-openai datasets
```

### Step 2: Configure PDF Path

Edit `run_ragas_evaluation.py` and update the PDF path:

```python
PDF_PATH = "c:/path/to/your/vision_transformer_paper.pdf"
```

### Step 3: Run RAG Pipeline

```bash
python -m backend.app.evaluation.run_ragas_evaluation
```

This will:
- Initialize your RAG pipeline with the Vision Transformer paper
- Run all 10 test questions through your RAG
- Collect generated answers and retrieved contexts
- Save results to `rag_results.json`

### Step 4: Evaluate with RAGAS

```bash
python -m backend.app.evaluation.evaluate_with_ragas
```

This will:
- Load the RAG results
- Evaluate using 4 RAGAS metrics
- Display overall scores and per-question analysis
- Save detailed report to `ragas_report.json`

## 📊 RAGAS Metrics Explained

| Metric | What it Measures | Good Score |
|--------|------------------|------------|
| **Faithfulness** | Factual consistency with retrieved context (no hallucinations) | > 0.75 |
| **Answer Relevancy** | How relevant the answer is to the question | > 0.75 |
| **Context Recall** | Did retrieval find all necessary information? | > 0.75 |
| **Context Precision** | Are retrieved contexts relevant and focused? | > 0.75 |

## 📁 Output Files

- `rag_results.json` - RAG pipeline outputs (answers + contexts)
- `ragas_report.json` - Detailed evaluation metrics and per-question scores

## 🎯 Interpreting Results

### Overall Scores
```
Faithfulness: 0.85 ✅ Good
Answer Relevancy: 0.65 ⚠️ Needs improvement
Context Precision: 0.78 ✅ Good
Context Recall: 0.55 ❌ Poor
```

### What to Do Next

**Low Faithfulness (< 0.75)** → Your RAG is hallucinating
- Improve answer generation prompt
- Add stricter grounding instructions
- Review context quality

**Low Answer Relevancy (< 0.75)** → Answers are off-topic
- Refine answer generation prompt
- Improve question reformulation
- Check if retrieval is finding relevant chunks

**Low Context Recall (< 0.75)** → Missing important information
- Increase retrieval `k` value (retrieve more chunks)
- Improve chunking strategy
- Check if important info is in the document

**Low Context Precision (< 0.75)** → Retrieving irrelevant content
- Improve reranking
- Adjust hybrid retriever weights
- Refine embedding model

## 🔄 Iteration Workflow

1. **Baseline**: Run evaluation, record scores
2. **Improve**: Adjust RAG components based on metrics
3. **Re-evaluate**: Run again and compare
4. **Repeat**: Keep iterating until all metrics > 0.75

## 📝 Test Dataset

The current test set contains 10 questions about the Vision Transformer paper covering:
- Architectural differences
- Technical details
- Performance metrics
- Specific results from tables

You can expand this by:
1. Adding more questions to the `test_data` dictionary
2. Creating separate test sets for different papers
3. Using the same evaluation scripts

## 🛠️ Customization

### Add More Questions

Edit `run_ragas_evaluation.py` and add to the `test_data` dictionary:

```python
test_data = {
    "question": [
        "Your new question here?",
        # ... more questions
    ],
    "ground_truth": [
        "Expected answer here",
        # ... more answers
    ],
    # ... rest of the structure
}
```

### Change Evaluator Model

Edit `evaluate_with_ragas.py`:

```python
evaluator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Cheaper option
```

### Evaluate Different Papers

1. Update `PDF_PATH` in `run_ragas_evaluation.py`
2. Create new test questions for that paper
3. Run the evaluation pipeline

## 💡 Tips

- **Start small**: Begin with 5-10 questions, then expand
- **Diverse questions**: Include factual, analytical, and table-based questions
- **Ground truth**: Manually verify answers for better evaluation
- **Track progress**: Save reports with timestamps to monitor improvements
- **Cost awareness**: RAGAS uses GPT-4 by default, which costs money per evaluation

## 🆘 Troubleshooting

**Error: PDF not found**
- Check the `PDF_PATH` is correct and file exists

**Error: RAGAS import failed**
- Run: `pip install ragas langchain-openai datasets`

**Error: OpenAI API key not found**
- Ensure your `.env` file has `OPENAI_API_KEY=your_key`

**Low scores across all metrics**
- Check if the PDF was processed correctly
- Verify the test questions match the paper content
- Review the generated answers in `rag_results.json`

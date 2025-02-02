# Comparison of Pretrained Text Classification Models using TOPSIS
## Overview
This project aims to identify the best pretrained model for **text classification** using the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** method. Several models are tested on various genres and then compared based on various performance and efficiency metrics to determine their ranking and suitability for deployment.

## Methodology
1. **Model Selection**: Various Hugging Face pretrained models are considered.
2. **Performance Evaluation**: Each model is tested on a benchmark dataset, and metrics such as accuracy, precision, recall, and F1-score are calculated.
3. **TOPSIS Ranking**: The models are ranked based on multiple criteria to identify the most optimal model.

## Models Evaluated
| Model Name | Hugging Face Model Checkpoint |
|------------|--------------------------------|
| DistilBERT | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Multilingual DistilBERT | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| Twitter RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment RoBERTa | `siebert/sentiment-roberta-large-english` |

## Evaluation Metrics
The following metrics are used to assess model performance and efficiency:
- **Accuracy**: Measures overall classification correctness.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to all actual positives.
- **F1-score**: Harmonic mean of precision and recall, balancing false positives and false negatives.
- **Hamming Loss**:  Fraction of incorrect labels to the total number of labels.
- **Log Loss**:  Measures the uncertainty of predictions by penalizing incorrect confidence levels.

## Results & Analysis
The results of the TOPSIS evaluation are summarized below:

## Model Performance Comparison for Education

| Model   | Accuracy  | Precision | Recall   | F1 Score  | Hamming Loss | Log Loss  | Topsis Score | Rank |
|---------|----------|-----------|----------|-----------|--------------|-----------|--------------|------|
| Model 1 | 0.457336 | 0.428517  | 0.377075 | 0.431859  | 0.568987     | 0.568987  | 0.272631     | 4    |
| Model 2 | 0.50307  | 0.420725  | 0.719871 | 0.59469   | 0.491398     | 0.491398  | 0.689287     | 1    |
| Model 3 | 0.50307  | 0.623297  | 0.274236 | 0.394842  | 0.491398     | 0.491398  | 0.313649     | 3    |
| Model 4 | 0.533559 | 0.500864  | 0.514193 | 0.551309  | 0.439672     | 0.439672  | 0.547393     | 2    |

## Model Performance Comparison for Sports

| Model   | Accuracy  | Precision | Recall   | F1 Score  | Hamming Loss | Log Loss  | Topsis Score | Rank |
|---------|----------|-----------|----------|-----------|--------------|-----------|--------------|------|
| Model 1 | 0.486453 | 0.474611  | 0.488678 | 0.486427  | 0.554964     | 0.554964  | 0.485037     | 3    |
| Model 2 | 0.465753 | 0.426020  | 0.529401 | 0.476775  | 0.678289     | 0.678289  | 0.482566     | 4    |
| Model 3 | 0.517503 | 0.573488  | 0.447955 | 0.508316  | 0.369976     | 0.369976  | 0.519938     | 1    |
| Model 4 | 0.527853 | 0.514162  | 0.529401 | 0.526962  | 0.308313     | 0.308313  | 0.514487     | 2    |

## Model Performance Comparison for Politics

| Model   | Accuracy  | Precision | Recall   | F1 Score  | Hamming Loss | Log Loss  | Topsis Score | Rank |
|---------|----------|-----------|----------|-----------|--------------|-----------|--------------|------|
| Model 1 | 0.543036 | 0.518298  | 0.532414 | 0.572134  | 0.278524     | 0.278524  | 0.612328     | 2    |
| Model 2 | 0.482699 | 0.422753  | 0.502835 | 0.505858  | 0.452602     | 0.452602  | 0.594085     | 3    |
| Model 3 | 0.349957 | 0.547093  | 0.029579 | 0.053790  | 0.835573     | 0.835573  | 0.381581     | 4    |
| Model 4 | 0.591306 | 0.503325  | 0.680307 | 0.643333  | 0.139262     | 0.139262  | 0.622231     | 1    |

## Model Performance Comparison for Finance

| Model   | Accuracy  | Precision | Recall   | F1 Score  | Hamming Loss | Log Loss  | Topsis Score | Rank |
|---------|----------|-----------|----------|-----------|--------------|-----------|--------------|------|
| Model 1 | 0.544033 | 0.489101  | 0.551999 | 0.558130  | 0.283052     | 0.283052  | 0.584253     | 2    |
| Model 2 | 0.516134 | 0.471633  | 0.532285 | 0.538197  | 0.345953     | 0.345953  | 0.581025     | 3    |
| Model 3 | 0.278991 | 0.541505  | 0.118285 | 0.194349  | 0.880608     | 0.880608  | 0.414452     | 4    |
| Model 4 | 0.599831 | 0.495090  | 0.630856 | 0.600885  | 0.157251     | 0.157251  | 0.586240     | 1    |

## Overall Performance
| Domain | Best Model | Model Name |
|-----------------|-----------------|-----------------|
| Education    | Model 2    | lxyuan/distilbert-base-multilingual-cased-sentiments-student    |
| Sports    | Model 3    | cardiffnlp/twitter-roberta-base-sentiment-latest   |
| Politics    | Model 4    | siebert/sentiment-roberta-large-english    |
| Finance    | Model 4    | siebert/sentiment-roberta-large-english    |


## Implementation & Usage
### **Prerequisites**
Ensure the following dependencies are installed before running the analysis:
```bash
pip install transformers scikit-learn numpy pandas matplotlib
```

### **Running the Notebook**
1. Open `comparison-of-pretrained-models-using-topsis.ipynb` in Jupyter Notebook.
2. Execute all the cells in sequence.
3. Review the model rankings and final results displayed in tables and plots.

## Conclusion
By applying **TOPSIS**, this project provides a structured approach to selecting the best **pretrained model** for text classification. The methodology balances multiple performance and efficiency factors, making it easier to choose the most suitable model for real-world applications.




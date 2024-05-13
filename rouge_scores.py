from rouge_score import rouge_scorer
import numpy as np  # Import numpy for averaging

def calculate_rouge_scores(references, hypotheses):
    # Initialize the RougeScorer with the desired metrics and a stemmer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    # Score each pair of reference and hypothesis
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores['rouge1'].append(score['rouge1'].fmeasure*10)
        scores['rouge2'].append(score['rouge2'].fmeasure*10)
        scores['rougeL'].append(score['rougeL'].fmeasure*10)

    # Calculate average scores using numpy to average the lists
    avg_scores = {key: np.mean(val) for key, val in scores.items()}

    return avg_scores
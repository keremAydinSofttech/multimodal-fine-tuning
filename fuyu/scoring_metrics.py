
import nltk
import json
import os
import pandas as pd
from rouge_score import rouge_scorer
from test_fuyu import *

def calculate_bleu_score(reference, hypothesis):
    reference = [reference.split()]  # Convert reference to a list of tokens
    hypothesis = hypothesis.split()   # Convert hypothesis to a list of tokens

    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)

    return bleu_score

def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer


def evaluate(model):

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    image_folder = '/mnt/keremaydin/data/images/val2014/'

    with open('/mnt/keremaydin/data/data_metrics.json', 'r') as f:
        data = json.load(f)

    columns_to_calculate = [
        'Existence',
        'Count',
        'Position',
        'Color',
        'OCR',
        'Commonsense_Reasoning',
        'Numerical_Calculation',
        'Text_Translation',
        'Poster',
        'Celebrity',
        'Scene',
        'Landmark',
        'Artwork'
    ]

        
    scores_df = pd.DataFrame(columns=['BLEU', 'ROUGE', 'WER'], index=columns_to_calculate)


    for i in range(len(data)):

        print(f'{round((i+1) / len(data) * 100, 2)}%')

        column = data[i]['classification']

        question = data[i]['conversations'][0]['value'].split('\n')[-1]

        image_id = data[i]['image_id']

        image_path = os.path.join(image_folder, image_id)

        answer = data[i]['conversations'][1]['value']

        prediction = model.generate_answer(question, image_path)

        bleu_score = calculate_bleu_score(answer, prediction)
        rouge_score = scorer.score(answer, prediction)['rouge1'].fmeasure
        wer_score = calculate_wer(answer, prediction)
        
        # BLEU
        scores_df.at[column, 'BLEU'] += bleu_score

        # ROUGE
        scores_df.at[column, 'ROUGE'] += rouge_score

        # WER
        scores_df.at[column, 'WER'] += wer_score


    return scores_df


if __name__ == '__main__':

    model_llava = ModelFuyu()

    scores = evaluate(model_llava)

    scores.to_excel('/mnt/keremaydin/fuyu/metric_results.xlsx', index=False)






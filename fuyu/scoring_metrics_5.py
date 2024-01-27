import nltk
import json
import os
import string
import pandas as pd
from rouge_score import rouge_scorer
import inflect
from test_fuyu import *
import warnings
warnings.filterwarnings("ignore")

def replace_numbers_with_text(input_string):
    p = inflect.engine()
    
    words = input_string.split()
    new_words = []
    
    for word in words:
        # Check if the word is a numeric value
        if word.isdigit():
            # Convert the numeric value to its text representation
            text_representation = p.number_to_words(word)
            new_words.append(text_representation)
        else:
            new_words.append(word)
    
    # Join the words back into a string
    output_string = ' '.join(new_words)
    
    return output_string

def calculate_bleu_score(reference, hypothesis):
    reference = [reference.split()]  # Convert reference to a list of tokens
    hypothesis = hypothesis.split()   # Convert hypothesis to a list of tokens

    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)

    return bleu_score

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

        
    scores_df = pd.DataFrame(columns=['BLEU', 'ROUGE_F1', 'ROUGE_RC'], index=columns_to_calculate)
    scores_df.fillna(0.0, inplace=True)

    translation_table = str.maketrans("", "", string.punctuation)

    data_split = len(data) // 4

    for i in range(data_split*3, len(data)):

        print(f'{round((i+1-data_split*3) / (data_split) * 100, 2)}%')

        column = data[i]['classification']

        question = data[i]['conversations'][0]['value'].split('\n')[-1]

        image_id = data[i]['image_id']

        image_path = os.path.join(image_folder, image_id)

        answer = data[i]['conversations'][1]['value']

        prediction = model.generate_answer(prompt= question, img_path=image_path)

        answer = replace_numbers_with_text(answer.lower().translate(translation_table))
        prediction = replace_numbers_with_text(prediction.lower().translate(translation_table))

        bleu_score = calculate_bleu_score(answer, prediction)
        rouge_score = scorer.score(answer, prediction)['rouge1']
        
        # BLEU
        scores_df.at[column, 'BLEU'] += bleu_score

        # ROUGE F1
        scores_df.at[column, 'ROUGE_F1'] += rouge_score.fmeasure

        # ROUGE RECALL
        scores_df.at[column, 'ROUGE_RC'] += rouge_score.recall


    return scores_df


if __name__ == '__main__':

    model_fuyu = ModelFuyu()

    scores = evaluate(model_fuyu)

    scores.to_excel('/mnt/keremaydin/fuyu/metric_results4.xlsx', index=False)











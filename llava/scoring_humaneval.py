import json
import os
import pandas as pd
import os
from test_llava import *

def humaneval(question, reference, candidate, img_path):

    print('Image path:', img_path)
    print('Question:', question)
    print('Correct:', reference)
    print('Prediction:', candidate)

    return int(input('Is it correct?\n'))


def evaluate(model_llava):

    image_folder = '/mnt/keremaydin/data/images/val2014/'

    with open('/mnt/keremaydin/data/data_humaneval.json', 'r') as f:
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

        
    scores_df = pd.DataFrame(columns=['Human-Eval'], index=columns_to_calculate)
    scores_df.fillna(0.0, inplace=True)


    for i in range(len(data)):

        print(f'{i+1}/{len(data)}')

        column = data[i]['classification']

        question = data[i]['conversations'][0]['value'].split('\n')[-1]

        image_id = data[i]['image_id']

        image_path = os.path.join(image_folder, image_id)

        answer = data[i]['conversations'][1]['value']

        prediction = model_llava.generate_answer(prompt= question, 
                                                 img_path=image_path)


        evaluation = humaneval(question, answer, prediction,image_path)

        scores_df.at[column, 'Human-Eval'] += evaluation
        

    return scores_df


if __name__ == '__main__':

    model_llava = ModelLLaVa()

    scores = evaluate(model_llava)

    scores.to_excel('/mnt/keremaydin/llava/humaneval_results.xlsx', index=False)











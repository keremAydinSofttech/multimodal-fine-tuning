import nltk
import json
import os
import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from test_llava import *

def correct_or_not(reference, candidate):
    '''
    Converts functions into swagger format.
    '''

    load_dotenv('/mnt/keremaydin/data/.env')

    prompt ="""
            Your task is to compare the correct answer and prediction in order to see if the prediction is accurate. If the prediction is correct write 'TRUE' if not write 'FALSE'.       
            <Example_1>:
            <Input>:
            answer: The table is made out of wood
            prediction: The table is made out of steel.
            </Input>
            <Output>:
            NO
            </Output>
            </Example_1>
            """
    
    prompt = prompt + '\n' + 'answer:' + reference + '\n' + 'prediction:' + candidate + '\n<Input>'

    client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version='2023-07-01-preview',
    azure_endpoint='https://softtech-openai-ailab.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-07-01-preview'
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
    )

    return chat_completion.choices[0].message.content


def evaluate(model_llava):

    image_folder = '/mnt/keremaydin/data/images/val2014/'

    with open('/mnt/keremaydin/data/data_selfeval.json', 'r') as f:
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

        
    scores_df = pd.DataFrame(columns=['Self-Eval'], index=columns_to_calculate)
    scores_df.fillna(0.0, inplace=True)


    for i in range(len(data)):

        print(f'{round((i+1) / len(data) * 100, 2)}%')

        column = data[i]['classification']

        question = data[i]['conversations'][0]['value'].split('\n')[-1]

        image_id = data[i]['image_id']

        image_path = os.path.join(image_folder, image_id)

        answer = data[i]['conversations'][1]['value']

        prediction = model_llava.generate_answer(prompt= question, 
                                                 img_path=image_path)


        evaluation = correct_or_not(answer, prediction)

        if evaluation.find('TRUE') != -1:
            scores_df.at[column, 'Self-Eval'] += 1
        

    return scores_df


if __name__ == '__main__':

    model_llava = ModelLLaVa()

    scores = evaluate(model_llava)

    scores.to_excel('/mnt/keremaydin/llava/selfeval_results.xlsx', index=False)











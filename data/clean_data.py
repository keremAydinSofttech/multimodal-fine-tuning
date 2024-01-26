from PIL import Image
import json
import os

with open('/mnt/keremaydin/data/data.json', 'r') as f:
    df = json.load(f)


image_folder = '/mnt/keremaydin/data/images'

new_df = []

for i in range(len(df)):

    print(f'{i+1}/{len(df)}')

    if 'image_id' in df[i]:

        try:
            img = Image.open(os.path.join(image_folder, df[i]['image_id']))
            new_df.append(df[i])
        except:
            if os.path.exists(os.path.join(image_folder, df[i]['image_id'])):
                os.remove(os.path.join(image_folder, df[i]['image_id'])) 

# Save the combined data to a JSON file
with open('data_clean.json', 'w') as outfile:
    json.dump(new_df, outfile)
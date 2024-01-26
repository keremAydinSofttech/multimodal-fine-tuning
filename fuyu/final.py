import pandas as pd
import json

with open('/mnt/keremaydin/data/data_clean.json', 'r') as f:
    data = json.load(f)

data = pd.DataFrame(data)

df = pd.read_csv('/mnt/keremaydin/fuyu/results0.csv')
df1 = pd.read_csv('/mnt/keremaydin/fuyu/results1.csv')
df2 = pd.read_csv('/mnt/keremaydin/fuyu/results2.csv')
df3 = pd.read_csv('/mnt/keremaydin/fuyu/results3.csv')
df4 = pd.read_csv('/mnt/keremaydin/fuyu/results4.csv')
df5 = pd.read_csv('/mnt/keremaydin/fuyu/results5.csv')

combined_df = pd.concat([df, df1, df2, df3, df4, df5], ignore_index=True)

sum_row = pd.DataFrame(combined_df.sum(), columns=['Sum']).transpose()

sum_row.to_csv('/mnt/keremaydin/fuyu/final_result.csv', index=False)





import os

import pandas as pd

"""
Select first question in 52 project of datasets, merge to 1 project 
"""
dir = "llm_perf/datasets/test/"
filenames = [
    os.path.join(dir, f)
    for f in os.listdir(dir)
    if os.path.isfile(os.path.join(dir, f)) and f.endswith(".csv")
]
# print(filenames)

rows = []

for filename in filenames:
    df = pd.read_csv(filename)
    rows.append(list(df.iloc[0]))
# print(rows)

result = pd.DataFrame(rows, columns=df.columns)

save_dir = "llm_perf/datasets/"
result.to_csv(f"{save_dir}/merged_52_test.csv", index=False)

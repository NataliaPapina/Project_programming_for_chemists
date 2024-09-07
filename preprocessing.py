import pandas as pd

data = pd.read_csv("nanotox_dataset.tsv", sep="\t", header='infer')

print(data.info())

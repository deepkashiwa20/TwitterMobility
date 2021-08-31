import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm


TWEET_DATASET_PATH = Path("data/typhoon_Hagibis_2019_dataset/typhoon_Hagibis_2019.pickle.zip")
RESULT_PATH = Path("data/typhoon_Hagibis_2019_dataset/daigo_bert_model_result.csv")
df = pd.read_pickle(TWEET_DATASET_PATH)
model_name = "daigo/bert-base-japanese-sentiment"
fn = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
with open(str(RESULT_PATH), "w") as f:
    f.write("TweetID,Sentiment,Score\n")
    for count, (idx, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
        rst = fn([row["CleanedText"]])
        f.write("{},{},{:.7f}\n".format(
            idx, "P" if rst[0]["label"] == "ポジティブ" else "N", rst[0]["score"]
        ))
        if count % 1000 == 0:
            f.flush()

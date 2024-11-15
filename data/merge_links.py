"""small script to standardize data labels and format"""

import pandas as pd

if __name__ == "__main__":
    all_df = pd.read_csv("links/raw/extra_data_links.csv").filter(
        items=["url", "status"], axis="columns")
    extra_non_phishing = all_df.loc[all_df["status"] == "legitimate"]
    extra_phishing = all_df.loc[all_df["status"] == "phishing"]

    # reading in the links, but specifying index column
    non_phishing = pd.read_csv("links/raw/non_phishing.csv",
                               index_col=0,
                               names=["url"])
    phishing = pd.read_csv("links/raw/phishing.csv", index_col=0, names=["url"])

    # adding status columns (this is the label)
    non_phishing["status"] = ["legitimate"] * len(non_phishing)
    phishing["status"] = ["phishing"] * len(phishing)

    # putting it all together
    all_data = pd.concat(
        [non_phishing, phishing, extra_non_phishing, extra_phishing])

    # have some empty values in the dataset for some reason
    all_data.dropna(inplace=True)
    all_data = all_data.sample(frac=1)  # shuffling the data

    print(f"{len(all_data)} links in the dataset")
    all_data.to_csv("links/all_data_links.csv", index=False)

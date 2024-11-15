# Phish-Sight-replication

This repository replicates and extends the work of Pandey and Mishra (2023) where they use the color palette of a website to understand if a given website is trying to phish or not.

Our direct replication of the work did not show the state-of-the-art results reported and we hypothesize that the original work did not preprocess the data carefully and thus predicted on 404 pages.

Our extension to the work is we add both a more careful preprocessing pipeline and NLP-based features based on the URL and the actual text of the website. The highest accuracy we achieve is 84% accuracy using a random forest on our new feature set, this is higher than any of the replication models.

## Preprocessing
![image6](https://github.com/user-attachments/assets/f8b85427-08d9-46b2-aee9-e4b42c7dfeb8)

## Replication Results (vs. reported vs. extension)
![image4](https://github.com/user-attachments/assets/0dfe18ce-5236-4b10-a205-516c2224b9e4)

## Usage
To use any of these files, please

`pip install -r requirements.txt`

## Contents
The directory is as follows:
- /data contains links and scraped data used for the research
- /metrics contains the metrics from our run of the repo
- /models contains pickled models for future use
- web_scraping.py is the script which does its namesake. It uses selenium and does a bulk of the feature extraction
- preprocessing.py cleans the data and continues a bit of the feature extraction
- classical_trainer.py tunes and trains the classical ML models and stores their results in metrics/.

## Contributors
Neh Majmudar (nehmajmudar@gmail.com)
Daniel Yakubov (danielyak@gmail.com)

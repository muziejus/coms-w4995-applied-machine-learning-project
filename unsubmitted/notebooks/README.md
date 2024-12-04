# Notebooks

This folder holds the various Jupyter notebooks that we have separately compiled.

- `text-prepare.ipynb` gives the necessary code to crawl through a folder of ProQuest TDMStudio XMLdocuments and extract important information as a csv. The features are:
    -  [`index`]: the article’s pandas index
    -  `goid`: the article’s global ProQuest ID
    -  `title`: the article’s headline
    -  `date`: the article’s date, in YYYY-MM-DD format
    -  `publisher`: the article’s publisher,
    -  `text`: the article’s full text, with HTML tags stripped out.


## Sample Notebooks

We also are putting sample notebooks from ProQuest, etc., here. 

- `AGAI_ChapGPT_Interaction_10_16.ipynb` is ProQuest’s sample notebook for using ChatGPT with TDMStudio. It relies on an API that Moacir has removed but has saved in an un-committed `.env` file in this project. It returns two csv files, one with the results of the prompt and one tracking token usage.


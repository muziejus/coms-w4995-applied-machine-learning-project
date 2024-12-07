# Using Sentiment Analysis to Forecast Share Prices

Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira

Throughout this project we have used ticker symbols to refer to companies:

Symbol | Company
-------|--------
DLTR   | Dollar Tree
LULU   | Lululemon
ULTA   | Ulta
WBA    | Walgreens
WMT    | Walmart

## .zip File Structure

- /
    - report.pdf - This is our final report describing the project.
    - main.py - This is the main Python module that executes the other modules, 
      which collect and analyze data. See below for more information.
    - utils/ - This folder holds some utility functions used repeatedly in 
      modules.
      - load_data.py - Loads the merged data for a specific company.
      - split_data.py - Splits the data into X and y and does a dev/test split
        where dev is the first 80% of the data, chronologically.
    - preprocessing/ - This folder holds preprocessing code.
        - external_indicators.py - This module captures our external economic 
          indicators.
        - share_prices.py - This module collects the share prices of our target 
          companies.
        - expand_financial_data.py - This module takes a company’s financial data 
          and expands to add rolling averages and Bollinger bands as well as target 
          prices and target classification (buy/sell).
        - sentiment.py - This module aggregates the article-by-article sentiment 
          analysis data into daily sentiment data for each company.
    - models/ - This folder holds the code for the actual machine learning model 
      training and evaluation.
    - plots/ - Some plots we made for our final analysis.
        - confusion_matrices.png - A plot of four confusion matrices,
          corresponding to the top-performing company for each of our four ML
          models.
    - data/ - We separated our data into financial data and sentiment data.
        - financial_data/
            - dltr.csv - DLTR historical data.
            - lulu.csv - LULU historical data.
            - ulta.csv - ULTA historical data.
            - wba.csv - WBA historical data.
            - wmt.csv - WMT historical data.
            - external_indicators.csv - External financial indicators.
            - merged_financial_data.csv - The above files merged together.
        - sentiment_data/ - Files generated by TDM Studio.
            - dltr_sent.parquet - DLTR sentiment analysis data.
            - lulu_sent.parquet - LULU sentiment analysis data.
            - ulta_sent.parquet - ULTA sentiment analysis data.
            - wba_sent.parquet - WBA sentiment analysis data.
            - wmt_sent.parquet - WMT sentiment analysis data.
        - model_metadata/ - Pickle files holding model metadata. These allow us
          to evaluate the models without retraining them.
            - lstm.h5 - LSTM model.
            - lstm_results.pkl - LSTM results.
            - random_forest_results.pkl - Random forest results.
            - svm_results.pkl - SVM results.
            - xgboost_result.pkl - XGBoost results.
        - dltr_merged_data.parquet - DLTR merged and aggregated data.
        - lulu_merged_data.parquet - LULU merged and aggregated data.
        - ulta_merged_data.parquet - ULTA merged and aggregated data.
        - wba_merged_data.parquet - WBA merged and aggregated data.
        - wmt_merged_data.parquet - WMT merged and aggregated data.
    - tdmstudio-notebooks/ - TDM Studio requires working in an isolated 
      environment, so we used notebooks to interact with it.
        - prepare-texts.ipynb - Gathers metadata about every article in five corpora.
          Most salient are article length and date. Filters out non-English articles.
          Creates 10,000 article long csv chunks to describe the corpus.
        - concatenate-corpora.ipynb - Combines the above csvs to prepare for 
          analysis. Also omits weekend articles and the top fifth of articles in 
          terms of length. Creates a single parquet file for each corpus.
        - analyze-texts.ipynb - Iterates over the corpus files above and saves,
          in 1,000 article chunks, parquet files that include sentiment scoring for
          each article. 
        - concatenate-analyses.ipynb - Combine data on sentiment analysis into files
          that end up being the sent.parquet files in data/sentiment_data.

## Code Description and Workflow

`main.py` takes up to two arguments from the command line:

1. Data collection is done with `python main.py collect_data`. This makes an attempt to redownload the financial and economic data and merges those with the sentiment data. It also zeroes out many of the NaNs in the sentiment data (a function of dropping all weekend news).
We have had difficulty getting the share price code to behave with the Yahoo! Finance API recently. In short, this step can be skipped.

2. The `eda` argument gives a brief description of the data for each company. 

3. For the individual models, there are two kinds of commands, one with the argument `rerun` as the second argument and the other without. With `rerun`, like `python main.py svm rerun`, the command will retrain and reevaluate the given model. Without does different things but typically nothing. The four model arguments are:
    - `lstm` for the LSTM model.
    - `random_forest` for the random forest model. If this is run without `rerun`, it will instead generate a plot and dataframe listing the more important features in the model.
    - `svm` for the SVM model.
    - `xgboost` for the XGBoost model. As with random forest, running this without `rerun` will generate a plot and dataframe listing the more important features in the model.

4. The `report` argument gives the accuracy and F1 scores for each company’s performance on each model. It then also provides a summary table describing the average performance for each model and the results of the top-performing company. Finally, it draws the confusion matrices for the top performing companies and saves the plot to `plots/confusion_matrices.png`.


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

## File Structure

- /
    - report.pdf - This is our final report describing the project.
    - main.py - This is the main Python module that executes the other modules, which collect and analyze data. See below for more information.
    - expand_financial_data.py - This module takes a company’s financial data and expands to add rolling averages and Bollinger bands as well as target prices and target classification (buy/sell).
    - external_indicators.py - This module captures our external economic indicators.
    - sentiment.py - This module aggregates the article-by-article sentiment analysis data into daily sentiment data for each company.
    - share_prices.py - This module collects the share prices of our target companies.
    - data/ - We separated our data into financial data and sentiment data.
        - financial_data/
            dltr.csv - DLTR historical data
            lulu.csv - LULU historical data
            ulta.csv - ULTA historical data
            wba.csv - WBA historical data
            wmt.csv - WMT historical data
            external_indicators.csv - External financial indicators
            merged_financial_data.csv - The above files merged together
        - sentiment_data/
            dltr_sent.parquet - DLTR sentiment analysis data
            lulu_sent.parquet - LULU sentiment analysis data
            ulta_sent.parquet - ULTA sentiment analysis data
            wba_sent.parquet - WBA sentiment analysis data
            wmt_sent.parquet - WMT sentiment analysis data
        - dltr_merged_data.parquet - DLTR merged and aggregated data
        - lulu_merged_data.parquet - LULU merged and aggregated data
        - ulta_merged_data.parquet - ULTA merged and aggregated data
        - wba_merged_data.parquet - WBA merged and aggregated data
        - wmt_merged_data.parquet - WMT merged and aggregated data
    - notebooks/ - Though most of our work was done in notebooks and only later converted to regular Python, the only notebooks we are including here are specific to the sentiment analysis in the TDMStudio sandbox.
        - sentiment_code/
            - prepare-texts.ipynb - 
            - concatenate-corpora.ipynb - Combine articles and prepare for batch processing
            - analyze-texts.ipynb - Execute sentiment analysis
            - concatenate-analyses.ipynb - Combine data on sentiment analysis

## Code Description and Workflow

`main.py` takes a single argument from the command line:

- `collect_data` rebuilds the data as available. If there are API issues, it falls back to already existing data files. All subsequent operations read the data anew from disk.
- `eda` gives a brief description of the data for each company. 
- `svm` executes our SVM analysis.


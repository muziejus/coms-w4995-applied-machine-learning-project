# Using Sentiment Analysis to Forecast Share Prices

Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira

Throughout we have used ticker symbols to refer to companies:

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
    - notebooks/ - We did the same with the code, as we split up into two teams.
        - financial_code/
            - external_indicators.ipynb - Gather external financial indicators
            - company_data.ipynb - Gather company financial information
            - merge_company_and_external_factors.ipynb - Merge the above together
        - sentiment_code/
            - prepare-texts.ipynb - 
            - concatenate-corpora.ipynb - Combine articles and prepare for batch processing
            - analyze-texts.ipynb - Execute sentiment analysis
            - concatenate-analyses.ipynb - Combine data on sentiment analysis

## Code Description and Workflow



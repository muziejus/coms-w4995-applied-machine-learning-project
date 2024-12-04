"""
# Expand Financial Data

Moacir P. de Sá Pereira

This module reads in the trading data for a company and returns an expanded
dataframe with the following columns:

Column | Type | Description
---|---|---
(index)| datetime | The date.
`open` | float | The opening price.
`high` | float | The high price.
`low` | float | The low price.
`close` | float | The closing price.
`volume` | float | The number of shares traded.
`dividends` | float | Dividends paid (if any).
`stock_splits` | float | Stock splits (if any).
`price_3d_rolling_mean` | float | Rolling three-day price mean.
`price_3d_rolling_std` | float | Rolling three-day price standard deviation.
`volume_3d_rolling_mean` | float | Rolling three-day volume mean.
`volume_3d_rolling_std` | float | Rolling three-day volume standard deviation.
`bollinger_3d_upper_band` | float | Bollinger three-day upper band.
`bollinger_3d_lower_band` | float | Bollinger three-day lower band.
`price_7d_rolling_mean` | float | Rolling seven-day price mean.
`price_7d_rolling_std` | float | Rolling seven-day price standard deviation.
`volume_7d_rolling_mean` | float | Rolling seven-day volume mean.
`volume_7d_rolling_std` | float | Rolling seven-day volume standard deviation.
`bollinger_7d_upper_band` | float | Bollinger seven-day upper band.
`bollinger_7d_lower_band` | float | Bollinger seven-day lower band.
`price_14d_rolling_mean` | float | Rolling 14-day price mean.
`price_14d_rolling_std` | float | Rolling 14-day price standard deviation.
`volume_14d_rolling_mean` | float | Rolling 14-day volume mean.
`volume_14d_rolling_std` | float | Rolling 14-day volume standard deviation.
`bollinger_14d_upper_band` | float | Bollinger 14-day upper band.
`bollinger_14d_lower_band` | float | Bollinger 14-day lower band.
`target_price` | float | Next day’s price.
`target` | int | Buy or sell recommendation (1 or 0)

"""

import pandas as pd


def expand_financial_data(df):
    new_column_names = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    df = df.rename(columns=new_column_names)

    # The share price data includes times and timezone offsets,
    # which get badly parse. We only need the date.
    df["date"] = df.date.apply(lambda x: x.split(" ")[0])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Ignore days with no closing price for calculating target and rolling values.
    business_days_df = df.copy()
    business_days_df = business_days_df.dropna(subset=["close"])
    business_days_df["target_price"] = business_days_df.close.shift(-1)
    business_days_df["target"] = (
        business_days_df.close.shift(-1) > business_days_df.close
    ).astype(int)

    for window in [3, 7, 14]:
        business_days_df[f"price_{window}d_rolling_mean"] = (
            business_days_df.close.rolling(window).mean()
        )
        business_days_df[f"price_{window}d_rolling_std"] = (
            business_days_df.close.rolling(window).std()
        )
        business_days_df[f"volume_{window}d_rolling_mean"] = (
            business_days_df.volume.rolling(window).mean()
        )
        business_days_df[f"volume_{window}d_rolling_std"] = (
            business_days_df.volume.rolling(window).std()
        )
        # Bollinger Bands
        std_dev = 2
        business_days_df[f"bollinger_{window}d_upper_band"] = business_days_df[
            f"price_{window}d_rolling_mean"
        ] + (business_days_df[f"price_{window}d_rolling_std"] * std_dev)
        business_days_df[f"bollinger_{window}d_lower_band"] = business_days_df[
            f"price_{window}d_rolling_mean"
        ] - (business_days_df[f"price_{window}d_rolling_std"] * std_dev)

    # Merge everything back together
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    business_days_df = business_days_df.reindex(full_date_range)

    df = df.merge(
        business_days_df[
            [
                "price_3d_rolling_mean",
                "price_3d_rolling_std",
                "volume_3d_rolling_mean",
                "volume_3d_rolling_std",
                "bollinger_3d_upper_band",
                "bollinger_3d_lower_band",
                "price_7d_rolling_mean",
                "price_7d_rolling_std",
                "volume_7d_rolling_mean",
                "volume_7d_rolling_std",
                "bollinger_7d_upper_band",
                "bollinger_7d_lower_band",
                "price_14d_rolling_mean",
                "price_14d_rolling_std",
                "volume_14d_rolling_mean",
                "volume_14d_rolling_std",
                "bollinger_14d_upper_band",
                "bollinger_14d_lower_band",
                "target",
                "target_price",
            ]
        ],
        how="left",
        left_index=True,
        right_index=True,
    )

    return df

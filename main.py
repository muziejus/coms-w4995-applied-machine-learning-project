# Using Sentiment Analysis to Forecast Share Prices
#
# Ruibin Lyu, Vi Mai, Julie Meunier, Tuba Opel, Moacir P. de Sá Pereira
from financial.external_indicators import save_external_indicators

start_date = "2019-01-01"
end_date = "2024-10-16"


save_external_indicators(start_date, end_date)

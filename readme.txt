The procedure makes use of a decomposable time series model with three main model components: trend, seasonality, and holidays.

y(t) = g(t) + s(t) + h(t) + e(t)

g(t) trend models non-periodic changes; linear or logistic
s(t) seasonality represents periodic changes; i.e. weekly, monthly, yearly
h(t) ties in effects of holidays; on potentially irregular schedules ≥ 1 day(s)
The error term e(t) represents any idiosyncratic changes which are not accommodated by the model; later we will make the parametric assumption that e(t) is normally distributed
## How to Setup

1. python -m venv .venv/
2. .venv\Scripts\activate
3. pip install -r requirements.txt

### Run Training Scripts
python -m src.pipeline.training_pipeline


#streamlit app

 streamlit run app.py
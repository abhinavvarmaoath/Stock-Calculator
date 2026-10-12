# Stock Calculator

A small Flask web app for stock trades. The calculator works out the real profit (or loss) on a trade after commissions and capital gains tax. The simulator runs thousands of random price paths to show the range of things that could happen if you hold a position for a while. The forecast scoreboard tests forecasting methods on a stock's past prices and shows whether any of them beat "the price stays put" by more than luck. The assistant is an AI agent you can ask about a trade in plain English.

**Author:** Abhinav Varma ([@abhinavvarmaoath](https://github.com/abhinavvarmaoath))

## Calculator

Fill in the details of a trade and the app gives you a report with:

- proceeds from the sale
- total cost (purchase price + commissions + tax)
- tax on the capital gain
- net profit
- return on investment (ROI)
- the sell price you need to break even

Inputs: stock symbol, number of shares, initial (buy) price, final (sell) price, buy commission, sell commission and the capital gains tax rate (%).

How it's worked out:

- proceeds = shares x final price
- gain = proceeds - purchase price - buy and sell commission
- tax = tax rate x gain (nothing if it's a loss, no refund is assumed)
- net profit = gain - tax
- ROI = net profit / total cost (purchase + commissions + tax)
- break-even price = initial price + total commission / shares

## Simulator

Enter a position and how long you'd hold it (1 month to 1 year). The simulator runs 10,000 random price paths and puts every one of them through the same maths as the calculator, so you get:

- the chance of making a profit after commissions and tax
- the expected and median net profit, the likely range (5th to 95th percentile) and the bad case (average of the worst 5% of runs)
- a chart of where the price could go: median, middle 50% and middle 90% of runs, and the break-even price
- a histogram of net profit, with losses in red and profits in blue
- a table view of both charts, and arrow key support on them

The price today and the volatility are worked out from the last 2 years of daily closes (yfinance, so it needs internet). If the lookup fails you can type them in under Advanced.

What it assumes:

- 252 trading days in a year
- expected yearly return is 0% unless you set it, so no view on which way the price goes
- normal random daily moves (geometric Brownian motion), or resampled past daily moves which keeps the big days a real stock has. Both are scaled to the same volatility
- commissions as entered, tax only on a gain
- it shows what could happen if prices behave like the model. It is not a prediction and not financial advice, and the numbers move a little each run because the runs are random

## Forecast scoreboard

Can anything beat "the price stays where it is"? Pick a stock and how far ahead to look (1 week, 1 month or 3 months) and the app tries six forecasting methods on its last 10 years of daily closes:

- the benchmark: today's price is the forecast
- drift: keeps growing at the average pace of the last year
- damped trend (Holt exponential smoothing)
- ARIMA(1,1,1)
- a ridge regression on recent returns, recent volatility and how far the price is from its 200 day average
- gradient boosting on the same inputs

Each method forecasts the price from about 70 past days and is marked against what really happened. You get:

- a verdict in plain words: did any method beat "price stays put" by more than luck can explain
- a score table: the average miss, how much better than the benchmark, how often it was nearer, how often it called up or down right, and a luck test
- a chart of those gaps, with the gap luck alone could have produced shaded in grey
- a check of the simulator's price range: how often the real price landed inside the 80% and 90% range the simulator would have drawn on those days

How it is kept honest:

- a forecast only ever sees prices up to the day it is made, and the methods that learn only learn from outcomes that were already known by then. A test hands every method a changed future and checks that nothing it says moves
- the luck test is a Diebold-Mariano test. Forecasts made closer together than the horizon share prices, so it counts them as fewer than they are, and the p values are adjusted for having tried several methods (Holm). In simulations with nothing but luck it flagged a difference 5 to 8% of the time, where 5% is the aim
- "can't tell" is not "no skill": the page says how big a gap had to be before it could have shown up, and for noisy methods at 3 months that can be more than any method could deliver
- the settings of every method were fixed in advance on made-up prices, never tuned on a stock's own history
- a forecast that fails is counted and shown, never dropped quietly or scored as a hit
- the luck test allows for the methods tried on one stock, not for how many stocks you look at: try 20 and about one will show a winner by luck alone
- it only looks backwards. There is no forecast price anywhere on the page, and a method that won on the past may not win tomorrow

Limits: a 1 week test needs 539 daily closes (about 2 years), a 1 month test 735 and a 3 month test 1,197, so a recent listing can only use the shorter ones. It takes 10 to 30 seconds. At most 2 run at once and one address gets 6 new ones every 10 minutes. The last 64 answers are remembered, so asking again for the same prices costs nothing. If the price lookup fails, or for something Yahoo does not have, paste your own closing prices under Advanced (oldest first, one on each line, up to 4,000).

## Assistant

Ask about a trade in plain English, like "I bought 100 shares of AAPL at $180 and paid $5 commission each way. With 15% tax, should I sell now or hold for 3 months?"

It is an AI agent that uses Claude through the Anthropic API. It decides which tools to call, runs them, reads what they give back and then answers. The tools are the same code the rest of the app uses:

- `get_stock_info` - latest close and yearly volatility for a ticker
- `calculate_trade` - the calculator
- `simulate_holding` - the simulator
- `forecast_scoreboard` - the forecast scoreboard, for when you ask whether a stock's price can be predicted or how far to trust the simulator's range

Every step is shown under the answer (the tool, what it was given and what it gave back), along with how many model calls and tokens it used.

How it is kept honest:

- the numbers in an answer have to come from a tool result or from what you wrote. The model is told never to do the maths or quote a price from memory
- every number the model sends to a tool is checked first (the same limits as the forms) and a bad one goes back as an error for the model to fix
- if a price lookup fails it says so and asks you for the numbers instead of guessing
- the scoreboard tool gives back how methods did on past prices and no forecast price, and the assistant is told never to predict a price itself
- a limit on model calls, on tool calls in one go and on the size of the chat stops one question running away
- nothing the model writes is ever treated as HTML

Set it up:

```
cp .env.example .env
```

Then put your key in `ANTHROPIC_API_KEY` and a model name from Anthropic's model list in `ANTHROPIC_MODEL`, and restart the app. `.env` is ignored by git, so the key stays on your machine.

It uses your API credits: each question is a few model calls. Answers are not financial advice.

If you put the app on the internet: the assistant spends your credits for anyone who can reach it, and a new forecast scoreboard keeps a processor busy for 10 seconds or more. They only limit each address (10 questions a minute, 6 new forecasts every 10 minutes), and behind a proxy every visitor can look like the same address. Put a login in front of it first.

## Run it

You need Python 3.10 or newer.

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8080 (set `FLASK_DEBUG=1` first if you want auto reload). The calculator, simulator and forecast scoreboard work without any key, only the assistant needs one.

## Tests

```
pytest
```

The tests take about a minute, because they fit real forecasting models. None of them use the internet or an API key. The assistant is tested against a small fake of the Anthropic API that rejects the same bad requests the real one does.

To check the assistant against the real model, add your key and model to `.env` and run `pytest -m live`. It costs a few cents.

## Files

- `app.py` - routes and input checks
- `calc.py` - the trade maths
- `sim.py` - the Monte Carlo simulation
- `forecast.py` - the forecasting methods and the walk-forward test that scores them
- `verdict.py` - the luck test, the adjustment for trying several methods and the simulator range check
- `prices.py` - price history lookup
- `limiter.py` - the per-address rate limit shared by the assistant and the scoreboard
- `agent.py` - the assistant: prompt, the tool loop, limits
- `tools.py` - the tools the assistant can use
- `templates/` - the pages
- `static/` - styles, the chart code (plain svg) and the chat page script
- `tests/` - pytest tests, including the fake API in `fakeapi.py`
- `.env.example` - the settings the assistant needs

## Roadmap

- watchlist alerts when a price crosses your break-even

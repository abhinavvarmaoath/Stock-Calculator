# Stock Calculator

A small Flask web app for stock trades. The calculator works out the real profit (or loss) on a trade after commissions and capital gains tax. The simulator runs thousands of random price paths to show the range of things that could happen if you hold a position for a while. The assistant is an AI agent you can ask about a trade in plain English.

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

## Assistant

Ask about a trade in plain English, like "I bought 100 shares of AAPL at $180 and paid $5 commission each way. With 15% tax, should I sell now or hold for 3 months?"

It is an AI agent that uses Claude through the Anthropic API. It decides which tools to call, runs them, reads what they give back and then answers. The tools are the same code the rest of the app uses:

- `get_stock_info` - latest close and yearly volatility for a ticker
- `calculate_trade` - the calculator
- `simulate_holding` - the simulator

Every step is shown under the answer (the tool, what it was given and what it gave back), along with how many model calls and tokens it used.

How it is kept honest:

- the numbers in an answer have to come from a tool result or from what you wrote. The model is told never to do the maths or quote a price from memory
- every number the model sends to a tool is checked first (the same limits as the forms) and a bad one goes back as an error for the model to fix
- if a price lookup fails it says so and asks you for the numbers instead of guessing
- a limit on model calls, on tool calls in one go and on the size of the chat stops one question running away
- nothing the model writes is ever treated as HTML

Set it up:

```
cp .env.example .env
```

Then put your key in `ANTHROPIC_API_KEY` and a model name from Anthropic's model list in `ANTHROPIC_MODEL`, and restart the app. `.env` is ignored by git, so the key stays on your machine.

It uses your API credits: each question is a few model calls. Answers are not financial advice.

If you put the app on the internet: the assistant spends your credits for anyone who can reach it. It only limits each address to 10 questions a minute, and behind a proxy every visitor can look like the same address. Put a login in front of it first.

## Run it

You need Python 3.10 or newer.

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8080 (set `FLASK_DEBUG=1` first if you want auto reload). The calculator and simulator work without any key, only the assistant needs one.

## Tests

```
pytest
```

None of the tests use the internet or an API key. The assistant is tested against a small fake of the Anthropic API that rejects the same bad requests the real one does.

To check the assistant against the real model, add your key and model to `.env` and run `pytest -m live`. It costs a few cents.

## Files

- `app.py` - routes and input checks
- `calc.py` - the trade maths
- `sim.py` - the Monte Carlo simulation
- `prices.py` - price history lookup
- `agent.py` - the assistant: prompt, the tool loop, limits
- `tools.py` - the tools the assistant can use
- `templates/` - the pages
- `static/` - styles, the chart code (plain svg) and the chat page script
- `tests/` - pytest tests, including the fake API in `fakeapi.py`
- `.env.example` - the settings the assistant needs

## Roadmap

- price forecasts with a backtest scoreboard
- watchlist alerts when a price crosses your break-even

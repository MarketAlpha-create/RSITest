import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def backtest():
    if request.method == 'POST':
        symbol = request.form['symbol']
        buy_level = int(request.form['buy_level'])
        sell_level = int(request.form['sell_level'])
        years = int(request.form['years'])

        end_date = pd.Timestamp.now().date()
        start_date = end_date - pd.Timedelta(days=years*365)

        data = yf.download(symbol, start=start_date, end=end_date)
        data['RSI'] = calculate_rsi(data['Close'], window=14)

        signals = generate_signals(data, buy_level, sell_level)
        returns = calculate_returns(data, signals)

        plot_results(returns)

        avg_return = returns.mean()
        cum_return = (returns + 1).cumprod()[-1] - 1

        return render_template('result.html', avg_return=avg_return, cum_return=cum_return)

    return render_template('index.html')

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(data, buy_level, sell_level):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0

    signals['Signal'][data['RSI'] < buy_level] = 1
    signals['Signal'][data['RSI'] > sell_level] = -1

    signals['Position'] = signals['Signal'].diff()
    return signals

def calculate_returns(data, signals):
    positions = signals['Position'].shift(1)
    returns = data['Close'].pct_change()
    strategy_returns = positions * returns
    return strategy_returns

def plot_results(returns):
    cum_returns = (returns + 1).cumprod() - 1
    plt.figure(figsize=(10, 6))
    plt.plot(cum_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True)
    plt.savefig('static/results.png')

if __name__ == '__main__':
    app.run()
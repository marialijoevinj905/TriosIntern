import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockTradingEnvironment:

    def __init__(self, data, window_size=10, initial_balance=10000):
        self.data = data.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.portfolio_values = []

        return self._get_state()

    def _get_state(self):
        price_window = self.data['Close'].iloc[self.current_step-self.window_size:self.current_step].values
        price_window = (price_window - price_window.mean()) / (price_window.std() + 1e-8)

        # Technical indicators
        current_price = self.data['Close'].iloc[self.current_step]
        sma_5 = self.data['SMA_5'].iloc[self.current_step]
        sma_20 = self.data['SMA_20'].iloc[self.current_step]
        rsi = self.data['RSI'].iloc[self.current_step]

        # Portfolio information (normalized)
        portfolio_info = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            self.net_worth / self.initial_balance
        ])

        # Technical indicators (normalized)
        tech_indicators = np.array([
            (current_price - sma_5) / current_price,
            (current_price - sma_20) / current_price,
            (rsi - 50) / 50  # RSI centered around 0
        ])

        return np.concatenate([price_window, tech_indicators, portfolio_info])

    def step(self, action):
        """Execute action and return next state, reward, done"""
        current_price = self.data['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth

        # Execute action
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trades.append(('SELL', self.current_step, current_price, self.shares_held))
                self.shares_held = 0

        # Update portfolio value
        self.net_worth = self.balance + self.shares_held * current_price
        self.portfolio_values.append(self.net_worth)

        # Calculate reward
        reward = self._calculate_reward(prev_net_worth, action)

        # Update max net worth for drawdown calculation
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        next_state = self._get_state() if not done else None

        return next_state, reward, done

    def _calculate_reward(self, prev_net_worth, action):
        """Calculate reward based on portfolio performance"""
        # Primary reward: portfolio value change
        portfolio_change = (self.net_worth - prev_net_worth) / prev_net_worth

        # Penalty for excessive trading (to encourage strategic trading)
        trade_penalty = -0.001 if action != 0 else 0

        # Bonus for maintaining upward trend
        trend_bonus = 0.001 if self.net_worth > self.max_net_worth * 0.95 else 0

        return portfolio_change + trade_penalty + trend_bonus

class DQNAgent:
    """
    Deep Q-Network Agent for Stock Trading
    """

    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []
        self.memory_size = 2000

        # Simple neural network weights (linear approximation)
        self.weights = np.random.randn(state_size, action_size) * 0.1
        self.bias = np.zeros(action_size)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        q_values = self._predict(state)
        return np.argmax(q_values)

    def _predict(self, state):
        """Predict Q-values for given state"""
        return np.dot(state, self.weights) + self.bias

    def replay(self, batch_size=32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        for i in batch:
            state, action, reward, next_state, done = self.memory[i]

            target = reward
            if not done and next_state is not None:
                target += 0.95 * np.max(self._predict(next_state))

            target_q = self._predict(state)
            target_q[action] = target

            # Simple gradient descent update
            error = target_q - self._predict(state)
            self.weights += self.learning_rate * np.outer(state, error)
            self.bias += self.learning_rate * error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def fetch_and_prepare_data(symbol="AAPL", period="2y"):
    """Fetch stock data and calculate technical indicators"""
    print(f"Fetching data for {symbol}...")

    # Fetch data
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)

    # Calculate technical indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()

    # RSI calculation
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)

    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

    # Drop rows with NaN values
    data = data.dropna()

    print(f"Data prepared: {len(data)} trading days")
    return data

def train_agent(data, episodes=100):
    """Train the DQN agent"""
    print("Training RL agent...")

    env = StockTradingEnvironment(data)
    state_size = len(env._get_state())
    agent = DQNAgent(state_size, 3)  # 3 actions: Hold, Buy, Sell

    episode_rewards = []
    portfolio_values = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.replay()
        episode_rewards.append(total_reward)
        portfolio_values.append(env.net_worth)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.4f}, "
                  f"Portfolio: ${env.net_worth:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent, episode_rewards, portfolio_values, env

def evaluate_agent(agent, data, initial_balance=10000):
    """Evaluate trained agent performance"""
    print("Evaluating trained agent...")

    env = StockTradingEnvironment(data, initial_balance=initial_balance)
    state = env.reset()

    # Disable exploration for evaluation
    agent.epsilon = 0

    portfolio_history = [initial_balance]
    actions_taken = []

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        portfolio_history.append(env.net_worth)
        actions_taken.append(action)
        state = next_state

        if done:
            break

    return env, portfolio_history, actions_taken

def create_comprehensive_analysis(data, agent, episode_rewards, portfolio_values,
                                eval_env, eval_portfolio, eval_actions, symbol="AAPL"):
    """Create comprehensive analysis with multiple visualizations"""

    fig = plt.figure(figsize=(20, 24))

    # 1. Stock Price and Technical Indicators
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
    ax1.plot(data.index, data['SMA_5'], label='SMA 5', alpha=0.7)
    ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    ax1.fill_between(data.index, data['BB_lower'], data['BB_upper'], alpha=0.2, label='Bollinger Bands')
    ax1.set_title(f'{symbol} Stock Price and Technical Indicators', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RSI and MACD
    ax2 = plt.subplot(4, 2, 2)
    ax2_twin = ax2.twinx()

    ax2.plot(data.index, data['RSI'], color='purple', label='RSI')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.set_ylabel('RSI', color='purple')
    ax2.set_ylim(0, 100)

    ax2_twin.plot(data.index, data['MACD'], color='blue', label='MACD')
    ax2_twin.plot(data.index, data['MACD_signal'], color='red', label='Signal')
    ax2_twin.set_ylabel('MACD', color='blue')

    ax2.set_title('RSI and MACD Indicators', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Training Progress
    ax3 = plt.subplot(4, 2, 3)
    episodes = range(1, len(episode_rewards) + 1)
    ax3.plot(episodes, episode_rewards, color='green', linewidth=2)
    ax3.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(episodes, episode_rewards, 1)
    p = np.poly1d(z)
    ax3.plot(episodes, p(episodes), "--", color='red', alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
    ax3.legend()

    # 4. Portfolio Value During Training
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(episodes, portfolio_values, color='orange', linewidth=2)
    ax4.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Balance')
    ax4.set_title('Portfolio Value During Training', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Agent Performance Evaluation
    ax5 = plt.subplot(4, 2, 5)

    # Plot stock price - ensure arrays have same length
    eval_dates = data.index[eval_env.window_size:eval_env.window_size + len(eval_portfolio) - 1]
    stock_performance = data['Close'].iloc[eval_env.window_size:eval_env.window_size + len(eval_portfolio) - 1].values
    stock_returns = (stock_performance / stock_performance[0]) * 10000

    # Ensure both arrays have the same length
    min_length = min(len(eval_dates), len(eval_portfolio) - 1, len(stock_returns))
    eval_dates = eval_dates[:min_length]
    stock_returns = stock_returns[:min_length]
    agent_portfolio = eval_portfolio[1:min_length + 1]

    ax5.plot(eval_dates, stock_returns, label='Buy & Hold', linewidth=2, alpha=0.7)
    ax5.plot(eval_dates, agent_portfolio, label='RL Agent', linewidth=2)
    ax5.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Balance')

    # Mark buy/sell actions
    buy_points = []
    sell_points = []
    for i, action in enumerate(eval_actions[:min_length]):
        if action == 1:  # Buy
            buy_points.append((eval_dates[i], agent_portfolio[i]))
        elif action == 2:  # Sell
            sell_points.append((eval_dates[i], agent_portfolio[i]))

    if buy_points:
        buy_x, buy_y = zip(*buy_points)
        ax5.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy', alpha=0.8)

    if sell_points:
        sell_x, sell_y = zip(*sell_points)
        ax5.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell', alpha=0.8)

    ax5.set_title('Agent Performance vs Buy & Hold', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Portfolio Value ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Action Distribution
    ax6 = plt.subplot(4, 2, 6)
    action_counts = [eval_actions.count(0), eval_actions.count(1), eval_actions.count(2)]
    action_labels = ['Hold', 'Buy', 'Sell']
    colors = ['blue', 'green', 'red']

    bars = ax6.bar(action_labels, action_counts, color=colors, alpha=0.7)
    ax6.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Actions')

    # Add value labels on bars
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(eval_actions)*100:.1f}%)',
                ha='center', va='bottom')

    # 7. Risk Analysis
    ax7 = plt.subplot(4, 2, 7)

    # Calculate daily returns with proper length matching
    portfolio_returns = np.diff(agent_portfolio) / agent_portfolio[:-1]
    stock_returns_daily = np.diff(stock_returns) / stock_returns[:-1]

    # Risk metrics
    portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
    stock_volatility = np.std(stock_returns_daily) * np.sqrt(252)

    # Plot return distributions
    ax7.hist(portfolio_returns, bins=30, alpha=0.7, label=f'RL Agent (σ={portfolio_volatility:.3f})', density=True)
    ax7.hist(stock_returns_daily, bins=30, alpha=0.7, label=f'Buy & Hold (σ={stock_volatility:.3f})', density=True)
    ax7.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Daily Returns')
    ax7.set_ylabel('Density')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Performance Metrics Summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')

    # Calculate performance metrics
    final_portfolio = agent_portfolio[-1]
    final_stock = stock_returns[-1]

    total_return_agent = (final_portfolio - 10000) / 10000 * 100
    total_return_stock = (final_stock - 10000) / 10000 * 100

    max_drawdown_agent = (max(agent_portfolio) - min(agent_portfolio)) / max(agent_portfolio) * 100
    max_drawdown_stock = (max(stock_returns) - min(stock_returns)) / max(stock_returns) * 100

    sharpe_agent = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
    sharpe_stock = np.mean(stock_returns_daily) / np.std(stock_returns_daily) * np.sqrt(252) if np.std(stock_returns_daily) > 0 else 0

    metrics_text = f"""
    Performance Summary

    RL Agent:
    • Total Return: {total_return_agent:.2f}%
    • Final Portfolio: ${final_portfolio:.2f}
    • Max Drawdown: {max_drawdown_agent:.2f}%
    • Sharpe Ratio: {sharpe_agent:.3f}
    • Volatility: {portfolio_volatility:.3f}
    • Total Trades: {len(eval_env.trades)}

    Buy & Hold:
    • Total Return: {total_return_stock:.2f}%
    • Final Portfolio: ${final_stock:.2f}
    • Max Drawdown: {max_drawdown_stock:.2f}%
    • Sharpe Ratio: {sharpe_stock:.3f}
    • Volatility: {stock_volatility:.3f}

    Agent vs Buy & Hold:
    • Excess Return: {total_return_agent - total_return_stock:.2f}%
    • Risk-Adjusted Performance: {'Better' if sharpe_agent > sharpe_stock else 'Worse'}
    """

    ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print detailed trade analysis
    print("\n" + "="*60)
    print("DETAILED TRADE ANALYSIS")
    print("="*60)

    if eval_env.trades:
        trade_df = pd.DataFrame(eval_env.trades, columns=['Action', 'Step', 'Price', 'Shares'])
        trade_df['Date'] = [data.index[eval_env.window_size + step] for step in trade_df['Step']]

        print("\nTrade History:")
        print(trade_df.to_string(index=False))

        # Calculate trade profitability
        buy_trades = trade_df[trade_df['Action'] == 'BUY']
        sell_trades = trade_df[trade_df['Action'] == 'SELL']

        if len(buy_trades) > 0 and len(sell_trades) > 0:
            print(f"\nTrade Statistics:")
            print(f"Total Buy Trades: {len(buy_trades)}")
            print(f"Total Sell Trades: {len(sell_trades)}")
            print(f"Average Buy Price: ${buy_trades['Price'].mean():.2f}")
            print(f"Average Sell Price: ${sell_trades['Price'].mean():.2f}")
    else:
        print("No trades executed during evaluation period.")

def main():
    """Main function to run the complete RL stock trading simulation"""

    print("="*60)
    print("REINFORCEMENT LEARNING STOCK TRADING SIMULATION")
    print("="*60)

    # Configuration
    SYMBOL = "AAPL"  # Change this to any stock symbol
    TRAINING_EPISODES = 100
    INITIAL_BALANCE = 10000

    try:
        # Step 1: Fetch and prepare data
        data = fetch_and_prepare_data(SYMBOL, period="2y")

        # Step 2: Train the agent
        agent, episode_rewards, portfolio_values, training_env = train_agent(data, TRAINING_EPISODES)

        # Step 3: Evaluate the trained agent
        eval_env, eval_portfolio, eval_actions = evaluate_agent(agent, data, INITIAL_BALANCE)

        # Step 4: Create comprehensive analysis
        create_comprehensive_analysis(data, agent, episode_rewards, portfolio_values,
                                    eval_env, eval_portfolio, eval_actions, SYMBOL)

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")

        # Final summary
        final_return = (eval_portfolio[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        print(f"\nFinal Results:")
        print(f"• Stock Symbol: {SYMBOL}")
        print(f"• Training Episodes: {TRAINING_EPISODES}")
        print(f"• Initial Balance: ${INITIAL_BALANCE:,}")
        print(f"• Final Portfolio Value: ${eval_portfolio[-1]:,.2f}")
        print(f"• Total Return: {final_return:.2f}%")
        print(f"• Total Trades Executed: {len(eval_env.trades)}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()

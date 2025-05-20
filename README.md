# Algorithmic Trading: A C++ Market Making Simulator 

C++ market simulation with a limit order book and adaptive market-making agent. Explores algorithmic trading concepts and dynamic quoting strategies. This project was undertaken for skill development in C++ and quantitative finance with a pdf accompanying the code.

This project was completed in September 2021 and has been drawn forward from a private repo into this one.

---

## Features 

* **Limit Order Book (LOB)**: Simulates a realistic LOB with price-time priority for order matching.
* **Market Maker Agent**: Implements an automated agent that provides liquidity by placing bid and ask orders.
* **Dynamic Quoting Strategy**: The agent adaptively adjusts its:
    * Bid-ask spread based on inventory levels.
    * Price skew to manage inventory risk (non-linear response).
    * Quoted quantity based on inventory deviation.
* **Market Simulation**:
    * Generates a synthetic asset price using an Ornstein-Uhlenbeck (mean-reverting) process.
    * Simulates random market participant orders (limit and market).
    * Applies price impact from executed trades.
* **Performance Analytics**: Calculates and outputs key performance indicators (KPIs) for the market maker agent, including P&L, Sharpe ratio, inventory levels, and average spread captured.
* **Configurable Parameters**: Market conditions and agent strategy parameters can be easily configured via a text file.
* **Data Logging**: Outputs simulation data (price history, agent P&L, inventory, quotes) to a CSV file for further analysis.

---

## Tech Stack 

* **C++17 (or newer)**: For the core simulation logic.
* **LaTeX**: For the project report.


             
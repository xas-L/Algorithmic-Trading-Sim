#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>

// Plotting can be done externally using the generated CSV file.

// Enumerations
enum class SignalType {
    BUY,
    SELL,
    HOLD
};

// Utility functions
namespace Utility {
    // Function to write data to a CSV file
    void writeToCSV(const std::string& filename, 
                    const std::vector<double>& prices, 
                    const std::vector<double>& shortSMA, 
                    const std::vector<double>& longSMA, 
                    const std::vector<SignalType>& signals,
                    const std::vector<double>& portfolioValues,
                    const std::vector<double>& dailyPnL, // Added for CSV
                    const std::vector<double>& drawdownSeries) { // Added for CSV

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        // Write header
        file << "Day,Price,ShortSMA,LongSMA,Signal,PortfolioValue,DailyPnL,DrawdownPercent\n";

        // Write data
        for (size_t i = 0; i < prices.size(); ++i) {
            file << i << "," << prices[i] << ",";

            // Write SMA values (if available)
            if (i < shortSMA.size() && shortSMA[i] != 0.0) file << shortSMA[i]; else file << "N/A"; // Assuming 0.0 is placeholder
            file << ",";

            if (i < longSMA.size() && longSMA[i] != 0.0) file << longSMA[i]; else file << "N/A"; // Assuming 0.0 is placeholder
            file << ",";

            // Write signal
            if (i < signals.size()) {
                switch (signals[i]) {
                    case SignalType::BUY: file << "BUY"; break;
                    case SignalType::SELL: file << "SELL"; break;
                    case SignalType::HOLD: file << "HOLD"; break;
                }
            } else {
                file << "N/A";
            }
            file << ",";

            // Write portfolio value
            if (i < portfolioValues.size()) file << portfolioValues[i]; else file << "N/A";
            file << ",";

            // Write Daily PnL
            if (i < dailyPnL.size()) file << dailyPnL[i]; else file << "N/A";
            file << ",";

            // Write Drawdown Percent
            if (i < drawdownSeries.size()) file << (drawdownSeries[i] * 100.0); else file << "N/A";
            file << "\n";
        }

        file.close();
    }

    // Function to calculate maximum drawdown
    double calculateMaxDrawdown(const std::vector<double>& portfolioValues) {
        if (portfolioValues.empty()) return 0.0;
        double maxDrawdown = 0.0;
        double peak = portfolioValues[0];

        for (const double& value : portfolioValues) {
            if (value > peak) {
                peak = value;
            }
            if (peak == 0) continue; // Avoid division by zero if peak is zero
            double drawdown = (peak - value) / peak;
            maxDrawdown = std::max(maxDrawdown, drawdown);
        }
        return maxDrawdown;
    }

    // Function to calculate drawdown series
    std::vector<double> calculateDrawdownSeries(const std::vector<double>& portfolioValues) {
        if (portfolioValues.empty()) return {};
        std::vector<double> drawdownSeries(portfolioValues.size());
        if (portfolioValues.empty()) return drawdownSeries; // Return empty if no values

        double peak = portfolioValues[0];

        for (size_t i = 0; i < portfolioValues.size(); ++i) {
            if (portfolioValues[i] > peak) {
                peak = portfolioValues[i];
            }
            if (peak == 0) { // Avoid division by zero
                 drawdownSeries[i] = 0.0;
            } else {
                drawdownSeries[i] = (peak - portfolioValues[i]) / peak;
            }
        }
        return drawdownSeries;
    }


    // Function to calculate Sharpe ratio
    double calculateSharpeRatio(const std::vector<double>& portfolioValues, double riskFreeRate) {
        if (portfolioValues.size() <= 1) {
            return 0.0;
        }

        std::vector<double> returns;
        for (size_t i = 1; i < portfolioValues.size(); ++i) {
            if (portfolioValues[i-1] == 0) { // Avoid division by zero
                returns.push_back(0.0);
            } else {
                returns.push_back((portfolioValues[i] - portfolioValues[i-1]) / portfolioValues[i-1]);
            }
        }

        if (returns.empty()) return 0.0;

        double meanReturn = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

        double sumSquaredDifferences = 0.0;
        for (const double& ret : returns) {
            sumSquaredDifferences += std::pow(ret - meanReturn, 2);
        }

        double stdDev = (returns.size() > 0) ? std::sqrt(sumSquaredDifferences / returns.size()) : 0.0;
        if (stdDev == 0) return 0.0; // Avoid division by zero if stdDev is zero (e.g. no change in portfolio)


        // Annualize (assuming daily returns)
        double annualizedReturn = meanReturn * 252;  // 252 trading days in a year
        double annualizedStdDev = stdDev * std::sqrt(252);

        if (annualizedStdDev == 0) return 0.0; // Avoid division by zero

        // Calculate Sharpe ratio
        return (annualizedReturn - riskFreeRate) / annualizedStdDev;
    }
}

// Class for simulating stock prices using Geometric Brownian Motion (GBM)
class StockSimulator {
private:
    std::mt19937 generator;

public:
    StockSimulator(unsigned int seed = std::random_device{}()) : generator(seed) {}

    // Function to generate a GBM path
    std::vector<double> generateGBMPath(double S0, double mu, double sigma, double T, int numSteps) {
        std::vector<double> pricePath(numSteps + 1);
        std::normal_distribution<double> normalDist(0.0, 1.0);

        double dt = T / numSteps;
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);

        pricePath[0] = S0;

        for (int i = 1; i <= numSteps; ++i) {
            double epsilon = normalDist(generator);
            pricePath[i] = pricePath[i-1] * std::exp(drift + diffusion * epsilon);
        }

        return pricePath;
    }
};

// Class for calculating Moving Averages
class MovingAverageCalculator {
public:
    // Function to calculate Simple Moving Average (SMA)
    std::vector<double> calculateSMA(const std::vector<double>& prices, int window) {
        size_t n = prices.size();
        std::vector<double> sma(n, 0.0); // Initialize with 0.0, will be N/A in CSV if 0.0 before window

        if (window <= 0 || n < static_cast<size_t>(window)) {
            return sma; 
        }

        double sum = 0.0;
        for (int i = 0; i < window; ++i) {
            sum += prices[i];
        }
        sma[window - 1] = sum / window;

        for (size_t i = window; i < n; ++i) {
            sum = sum - prices[i - window] + prices[i];
            sma[i] = sum / window;
        }
        return sma;
    }
};

// Class for implementing the trading strategy
class TradingStrategy {
private:
    int shortWindow;
    int longWindow;

public:
    TradingStrategy(int shortWindow, int longWindow)
        : shortWindow(shortWindow), longWindow(longWindow) {}

    // Function to generate trading signals based on SMA crossover
    std::pair<std::vector<SignalType>, std::pair<std::vector<double>, std::vector<double>>> 
    generateSignals(const std::vector<double>& pricePath) {
        MovingAverageCalculator maCalculator;
        std::vector<double> shortSMA_full = maCalculator.calculateSMA(pricePath, shortWindow);
        std::vector<double> longSMA_full = maCalculator.calculateSMA(pricePath, longWindow);

        size_t n = pricePath.size();
        std::vector<SignalType> signals(n, SignalType::HOLD);

        for (size_t i = std::max(static_cast<size_t>(longWindow), static_cast<size_t>(1)); i < n; ++i) {
            // Ensure SMAs at i and i-1 are valid (not the initial 0.0s before window is filled)
            // Check if SMAs are non-zero, assuming 0.0 is the placeholder for not-yet-calculated SMAs.
            bool shortSmaValidPrev = (i - 1 >= static_cast<size_t>(shortWindow - 1)) && shortSMA_full[i-1] != 0.0;
            bool shortSmaValidCurr = (i >= static_cast<size_t>(shortWindow - 1)) && shortSMA_full[i] != 0.0;
            bool longSmaValidPrev = (i - 1 >= static_cast<size_t>(longWindow - 1)) && longSMA_full[i-1] != 0.0;
            bool longSmaValidCurr = (i >= static_cast<size_t>(longWindow - 1)) && longSMA_full[i] != 0.0;

            if (!(shortSmaValidPrev && shortSmaValidCurr && longSmaValidPrev && longSmaValidCurr)) {
                // If any SMA is not valid (still the initial 0.0 placeholder), skip this iteration.
                continue;
            }

            if (shortSMA_full[i-1] <= longSMA_full[i-1] && shortSMA_full[i] > longSMA_full[i]) {
                signals[i] = SignalType::BUY;
            } else if (shortSMA_full[i-1] >= longSMA_full[i-1] && shortSMA_full[i] < longSMA_full[i]) {
                signals[i] = SignalType::SELL;
            }
        }
        return std::make_pair(signals, std::make_pair(shortSMA_full, longSMA_full));
    }
};

// Class for backtesting the trading strategy
class Backtester {
public:
    struct BacktestResult {
        std::vector<double> portfolioValues;
        std::vector<double> dailyPnL; 
        std::vector<double> drawdownSeries; 
        double totalReturn;
        double sharpeRatio;
        double maxDrawdownValue; 
        int numTrades;
    };

    // Function to run backtest
    BacktestResult runBacktest(const std::vector<double>& pricePath, 
                               const std::vector<SignalType>& signals, 
                               double initialCash) {
        BacktestResult result;
        size_t n = pricePath.size();
        if (n == 0) return result; // Handle empty price path

        result.portfolioValues.resize(n, 0.0);
        result.dailyPnL.resize(n, 0.0); 

        double cash = initialCash;
        double shares = 0.0;
        int numTrades = 0;

        if (n > 0) {
            result.portfolioValues[0] = initialCash;
            result.dailyPnL[0] = 0.0; 
        }


        for (size_t i = 1; i < n; ++i) {
            if (pricePath[i] <= 0) { // Skip if price is not positive
                result.portfolioValues[i] = result.portfolioValues[i-1]; // Carry over portfolio value
                result.dailyPnL[i] = 0; // No PnL change
                continue;
            }

            if (signals[i] == SignalType::BUY && shares == 0.0) { 
                shares = cash / pricePath[i];
                cash = 0.0;
                numTrades++;
            } else if (signals[i] == SignalType::SELL && shares > 0.0) { 
                cash = shares * pricePath[i];
                shares = 0.0;
                numTrades++;
            }

            result.portfolioValues[i] = cash + shares * pricePath[i];
            result.dailyPnL[i] = result.portfolioValues[i] - result.portfolioValues[i-1];
        }

        if (shares > 0.0 && !pricePath.empty() && pricePath.back() > 0) {
            cash = shares * pricePath.back();
            // shares = 0.0; // Not strictly needed for final value calculation
            result.portfolioValues.back() = cash; 
            if (n > 1) { 
                 result.dailyPnL.back() = result.portfolioValues.back() - result.portfolioValues[n-2];
            } else if (n==1) { // Edge case: only one day, liquidation changes PnL from 0
                 result.dailyPnL.back() = result.portfolioValues.back() - initialCash;
            }
        }


        if (initialCash == 0) { 
            result.totalReturn = (result.portfolioValues.empty() || result.portfolioValues.back() == 0) ? 0.0 : 1.0;
        } else {
            result.totalReturn = result.portfolioValues.empty() ? 0.0 : (result.portfolioValues.back() - initialCash) / initialCash;
        }
        result.sharpeRatio = Utility::calculateSharpeRatio(result.portfolioValues, 0.02);  
        result.drawdownSeries = Utility::calculateDrawdownSeries(result.portfolioValues);
        result.maxDrawdownValue = Utility::calculateMaxDrawdown(result.portfolioValues); 
        result.numTrades = numTrades;

        return result;
    }
};


int main() {
    // Parameters
    double S0 = 100.0;           // Initial stock price
    double mu = 0.08;            // Annual drift
    double sigma = 0.20;         // Annual volatility
    double T = 1.0;              // Time horizon (1 year)
    int numSteps = 252;          // Number of trading days
    double initialCash = 100000.0; // Initial cash
    int shortWindow = 10;        // Short-term SMA window
    int longWindow = 30;         // Long-term SMA window

    // Create simulation
    StockSimulator simulator;
    std::vector<double> pricePath = simulator.generateGBMPath(S0, mu, sigma, T, numSteps);

    // Generate trading signals
    TradingStrategy strategy(shortWindow, longWindow);
    auto result_pair = strategy.generateSignals(pricePath);
    std::vector<SignalType> signals = result_pair.first;
    std::vector<double> shortSMA = result_pair.second.first;
    std::vector<double> longSMA = result_pair.second.second;


    // Run backtest
    Backtester backtester;
    Backtester::BacktestResult result = backtester.runBacktest(pricePath, signals, initialCash);

    // Output results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========== SMA Crossover Strategy Backtest Results ==========" << std::endl;
    if (!result.portfolioValues.empty()) {
        std::cout << "Initial Portfolio Value: $" << initialCash << std::endl;
        std::cout << "Final Portfolio Value: $" << result.portfolioValues.back() << std::endl;
    } else {
        std::cout << "Initial Portfolio Value: $" << initialCash << std::endl;
        std::cout << "Final Portfolio Value: N/A (No results)" << std::endl;
    }
    std::cout << "Total Return: " << (result.totalReturn * 100) << "%" << std::endl;
    std::cout << "Number of Trades: " << result.numTrades << std::endl;
    std::cout << "Sharpe Ratio: " << result.sharpeRatio << std::endl;
    std::cout << "Maximum Drawdown: " << (result.maxDrawdownValue * 100) << "%" << std::endl;

    // Write results to CSV file for further analysis/plotting
    if (!pricePath.empty()){ // Ensure there's data to write
        Utility::writeToCSV("sma_crossover_backtest.csv", 
                            pricePath, 
                            shortSMA, 
                            longSMA, 
                            signals, 
                            result.portfolioValues,
                            result.dailyPnL,
                            result.drawdownSeries);
        std::cout << "\nResults saved to 'sma_crossover_backtest.csv'" << std::endl;
        std::cout << "You can use a Python script with Matplotlib or other tools " << std::endl;
        std::cout << "to read 'sma_crossover_backtest.csv' and generate plots." << std::endl;
    } else {
        std::cout << "\nNo data to save to CSV." << std::endl;
    }

    return 0;
}

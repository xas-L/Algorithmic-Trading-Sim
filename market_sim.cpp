#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>
#include <limits> // Required for std::numeric_limits
#include <memory> // For std::unique_ptr, std::shared_ptr 
#include <functional> // For std::greater
#include <set> // <--- Add this line

// Forward Declarations 
class OrderBook;
class MarketMakerAgent;
class MarketSimulator;

// ----------------------------------------------------------------------------
// Content from Order.h
// ----------------------------------------------------------------------------
enum class OrderSide {
    BID,
    ASK
};

enum class OrderType {
    LIMIT,
    MARKET
};

struct Order {
    double price;
    int quantity;
    OrderSide side;
    OrderType type;
    std::int64_t timestamp;  // milliseconds since epoch
    std::string order_id;

    // Constructor
    Order(double p, int qty, OrderSide s, OrderType ot, const std::string& id)
        : price(p), quantity(qty), side(s), type(ot), order_id(id) {
        auto now = std::chrono::system_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }

    // Constructor with explicit timestamp (for testing or specific scenarios)
    Order(double p, int qty, OrderSide s, OrderType ot,
          const std::string& id, std::int64_t ts)
        : price(p), quantity(qty), side(s), type(ot),
          timestamp(ts), order_id(id) {}
};

struct Trade {
    std::string maker_order_id;
    std::string taker_order_id;
    double price;
    int quantity;
    std::int64_t timestamp;

    Trade(const std::string& maker_id, const std::string& taker_id,
          double p, int qty)
        : maker_order_id(maker_id), taker_order_id(taker_id),
          price(p), quantity(qty) {
        auto now = std::chrono::system_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }
};


// ----------------------------------------------------------------------------
// Content from OrderBook.h (Class Definition)
// ----------------------------------------------------------------------------
class OrderBook {
public:
    OrderBook();

    void add_order(const Order& order);
    bool cancel_order(const std::string& order_id);
    std::vector<Trade> match_market_order(const Order& market_order);
    std::vector<Trade> match_limit_orders();
    std::pair<double, double> get_BBO() const;
    int get_depth_at_price(double price, OrderSide side) const;
    int get_total_volume(OrderSide side) const;
    bool has_order(const std::string& order_id) const;

    const std::map<double, std::vector<Order>, std::greater<double>>& get_bids() const { return bids_; }
    const std::map<double, std::vector<Order>>& get_asks() const { return asks_; }

    void register_agent(MarketMakerAgent* agent);

private:
    std::map<double, std::vector<Order>, std::greater<double>> bids_;
    std::map<double, std::vector<Order>> asks_;
    std::map<std::string, std::pair<double, size_t>> order_locations_;
    MarketMakerAgent* mm_agent_ = nullptr;

    void notify_agent_of_fill(const Order& order, double fill_price, int fill_quantity);
};

// ----------------------------------------------------------------------------
// Content from MarketMakerAgent.h (Class Definition)
// ----------------------------------------------------------------------------
class MarketMakerAgent {
public:
    MarketMakerAgent(
        OrderBook& order_book,
        double initial_capital,
        int target_inventory,
        int max_inventory,
        double base_spread,
        double inventory_skew_factor,
        int quote_quantity
    );

    void update_quotes(double fair_value);
    void on_fill(const Order& filled_order, double fill_price, int fill_quantity);
    double get_pnl() const { return realized_pnl_ + unrealized_pnl_; }
    double get_realized_pnl() const { return realized_pnl_; }
    double get_unrealized_pnl() const { return unrealized_pnl_; }
    int get_inventory() const { return inventory_; }
    void update_unrealized_pnl(double fair_value);

    void get_stats(
        double& total_pnl, double& avg_pnl_per_trade,
        double& stddev_pnl, double& sharpe_ratio,
        int& max_inventory, int& total_trades,
        double& avg_spread_captured
    ) const;

    std::pair<double, double> get_current_quotes() const { return {current_bid_, current_ask_}; }
    const std::vector<std::pair<std::int64_t, double>>& get_pnl_history() const { return pnl_history_; }
    const std::vector<std::pair<std::int64_t, int>>& get_inventory_history() const { return inventory_history_; }
    const std::vector<std::pair<std::int64_t, std::pair<double, double>>>& get_quote_history() const { return quote_history_; }

private:
    OrderBook& order_book_;
    double capital_; // Now used in accounting calculations
    int inventory_;
    int target_inventory_;
    int max_inventory_;
    double realized_pnl_;
    double unrealized_pnl_;
    double base_spread_;
    double inventory_skew_factor_;
    int quote_quantity_;
    double avg_entry_price_;
    std::string current_bid_id_;
    std::string current_ask_id_;
    double current_bid_;
    double current_ask_;
    int total_trades_;
    double total_volume_; // Changed from int to double to match MarketMakerAgent.cpp
    double sum_pnl_per_trade_;
    double sum_squared_pnl_per_trade_;
    int max_inventory_held_;
    double sum_spread_captured_;
    std::vector<std::pair<std::int64_t, double>> pnl_history_;
    std::vector<std::pair<std::int64_t, int>> inventory_history_;
    std::vector<std::pair<std::int64_t, std::pair<double, double>>> quote_history_;

    void cancel_existing_orders();
    void place_quotes(double fair_value);
    std::string generate_order_id() const;
    void record_state();
};

// ----------------------------------------------------------------------------
// Content from MarketSimulator.h (Class Definition)
// ----------------------------------------------------------------------------
class MarketSimulator {
public:
    MarketSimulator(
        OrderBook& order_book,
        MarketMakerAgent& agent,
        double initial_price,
        double price_volatility,
        double mean_reversion_strength,
        double order_arrival_rate,
        double market_order_probability,
        double price_impact_factor,
        int min_quantity,
        int max_quantity
    );

    void run_simulation_step();
    void run_simulation(int num_steps);
    double get_current_price() const { return current_price_; }
    const std::vector<std::pair<std::int64_t, double>>& get_price_history() const { return price_history_; }
    const std::vector<Trade>& get_trade_history() const { return trade_history_; }

private:
    OrderBook& order_book_;
    MarketMakerAgent& agent_;
    double current_price_;
    double price_volatility_;
    double mean_reversion_strength_;
    double initial_price_; // Added this member based on constructor usage
    double order_arrival_rate_;
    double market_order_probability_;
    double price_impact_factor_;
    int min_quantity_;
    int max_quantity_;
    std::mt19937 rng_;
    std::normal_distribution<double> price_shock_dist_;
    // std::exponential_distribution<double> arrival_time_dist_; // Not used in provided MarketSimulator.cpp
    std::uniform_real_distribution<double> uniform_dist_;
    std::uniform_int_distribution<int> quantity_dist_;
    std::vector<std::pair<std::int64_t, double>> price_history_;
    std::vector<Trade> trade_history_;

    void generate_random_order();
    void update_price_model();
    std::string generate_participant_id() const;
};


// ----------------------------------------------------------------------------
// Content from OrderBook.cpp (Method Implementations)
// ----------------------------------------------------------------------------
OrderBook::OrderBook() : mm_agent_(nullptr) {}

void OrderBook::add_order(const Order& order) {
    if (order.type == OrderType::MARKET) {
        // Market orders should be handled by match_market_order
        // Potentially log this or throw an exception if it's an invalid state
        std::cerr << "Warning: Attempted to add a MARKET order directly to the book. Order ID: " << order.order_id << std::endl;
        // For now, we'll allow it but ideally, this path shouldn't be hit if logic is correct.
        // Or, throw std::invalid_argument("Cannot add MARKET orders directly to the book via add_order.");
    }

    // Fix: Use separate handling for bid and ask sides instead of conditional operator
    if (order.side == OrderSide::BID) {
        auto& price_level = bids_[order.price];
        size_t position = price_level.size();
        price_level.push_back(order);
        order_locations_[order.order_id] = {order.price, position};
    } else { // ASK
        auto& price_level = asks_[order.price];
        size_t position = price_level.size();
        price_level.push_back(order);
        order_locations_[order.order_id] = {order.price, position};
    }

    match_limit_orders(); // Check for immediate matches
}

bool OrderBook::cancel_order(const std::string& order_id) {
    auto it = order_locations_.find(order_id);
    if (it == order_locations_.end()) {
        return false; // Order not found
    }

    double price = it->second.first;
    size_t position_in_level = it->second.second; // Renamed for clarity

    // Determine which book (bids_ or asks_) the order belongs to.
    // This requires knowing the side of the order. We don't store side in order_locations_.
    // We need to iterate to find it or store side. For simplicity, we check both.

    bool found_in_bids = false;
    if (bids_.count(price)) {
        auto& price_level = bids_.at(price);
        if (position_in_level < price_level.size() && price_level[position_in_level].order_id == order_id) {
            found_in_bids = true;
            if (position_in_level < price_level.size() - 1) {
                std::swap(price_level[position_in_level], price_level.back());
                order_locations_[price_level[position_in_level].order_id] = {price, position_in_level}; // Update swapped order's location
            }
            price_level.pop_back();
            if (price_level.empty()) {
                bids_.erase(price);
            }
        }
    }

    if (!found_in_bids && asks_.count(price)) {
         auto& price_level = asks_.at(price);
        if (position_in_level < price_level.size() && price_level[position_in_level].order_id == order_id) {
            if (position_in_level < price_level.size() - 1) {
                std::swap(price_level[position_in_level], price_level.back());
                order_locations_[price_level[position_in_level].order_id] = {price, position_in_level}; // Update swapped order's location
            }
            price_level.pop_back();
            if (price_level.empty()) {
                asks_.erase(price);
            }
        } else if (!found_in_bids) { // If not found in bids and not correctly identified in asks
             order_locations_.erase(it); // Clean up inconsistent location
             return false; // Order ID was in locations, but not found at that position in any book.
        }
    } else if (!found_in_bids) { // Not in bids and price level doesn't exist in asks
        order_locations_.erase(it);
        return false;
    }

    order_locations_.erase(it);
    return true;
}


std::vector<Trade> OrderBook::match_market_order(const Order& market_order) {
    if (market_order.type != OrderType::MARKET) {
        throw std::invalid_argument("match_market_order called with non-MARKET order.");
    }

    std::vector<Trade> trades;
    int remaining_quantity = market_order.quantity;

    if (market_order.side == OrderSide::BID) { // Market buy order, match against asks_
        auto ask_it = asks_.begin();
        while (remaining_quantity > 0 && ask_it != asks_.end()) {
            auto& price_level = ask_it->second;
            double trade_price = ask_it->first;

            auto order_it = price_level.begin();
            while (remaining_quantity > 0 && order_it != price_level.end()) {
                Order& resting_order = *order_it;
                int fill_quantity = std::min(remaining_quantity, resting_order.quantity);

                trades.emplace_back(resting_order.order_id, market_order.order_id, trade_price, fill_quantity);
                notify_agent_of_fill(resting_order, trade_price, fill_quantity); // Notify for resting order
                // Market orders don't usually have an agent to notify in this simplified model for the taker side.

                resting_order.quantity -= fill_quantity;
                remaining_quantity -= fill_quantity;

                if (resting_order.quantity <= 0) {
                    order_locations_.erase(resting_order.order_id);
                    order_it = price_level.erase(order_it); // Erase and move to next
                } else {
                    ++order_it;
                }
            }
            if (price_level.empty()) {
                ask_it = asks_.erase(ask_it); // Erase and move to next
            } else {
                ++ask_it;
            }
        }
    } else { // Market sell order, match against bids_
        auto bid_it = bids_.begin();
        while (remaining_quantity > 0 && bid_it != bids_.end()) {
            auto& price_level = bid_it->second;
            double trade_price = bid_it->first;

            auto order_it = price_level.begin();
            while (remaining_quantity > 0 && order_it != price_level.end()) {
                Order& resting_order = *order_it;
                int fill_quantity = std::min(remaining_quantity, resting_order.quantity);

                trades.emplace_back(resting_order.order_id, market_order.order_id, trade_price, fill_quantity);
                notify_agent_of_fill(resting_order, trade_price, fill_quantity);

                resting_order.quantity -= fill_quantity;
                remaining_quantity -= fill_quantity;

                if (resting_order.quantity <= 0) {
                    order_locations_.erase(resting_order.order_id);
                    order_it = price_level.erase(order_it);
                } else {
                    ++order_it;
                }
            }
            if (price_level.empty()) {
                bid_it = bids_.erase(bid_it);
            } else {
                ++bid_it;
            }
        }
    }
    return trades;
}

std::vector<Trade> OrderBook::match_limit_orders() {
    std::vector<Trade> trades;
    while (!bids_.empty() && !asks_.empty()) {
        double best_bid_price = bids_.begin()->first;
        auto& best_bid_level = bids_.begin()->second;
        double best_ask_price = asks_.begin()->first;
        auto& best_ask_level = asks_.begin()->second;

        if (best_bid_price < best_ask_price) {
            break; // No crossing orders
        }

        // Orders cross or touch. Determine trade price (aggressor is the newer order)
        Order& bid_order = best_bid_level.front();
        Order& ask_order = best_ask_level.front();

        double trade_price;
        // Price-time priority: existing order sets the price if they cross.
        // If bid_order is older, it was resting, ask_order is aggressor, trade at bid_order.price
        // If ask_order is older, it was resting, bid_order is aggressor, trade at ask_order.price
        if (bid_order.timestamp < ask_order.timestamp) { // bid is older (resting), ask is aggressor
            trade_price = bid_order.price; 
        } else { // ask is older (resting) or same timestamp, bid is aggressor
            trade_price = ask_order.price;
        }


        int fill_quantity = std::min(bid_order.quantity, ask_order.quantity);

        trades.emplace_back(bid_order.order_id, ask_order.order_id, trade_price, fill_quantity);
        notify_agent_of_fill(bid_order, trade_price, fill_quantity);
        notify_agent_of_fill(ask_order, trade_price, fill_quantity);

        bid_order.quantity -= fill_quantity;
        ask_order.quantity -= fill_quantity;

        if (bid_order.quantity <= 0) {
            order_locations_.erase(bid_order.order_id);
            best_bid_level.erase(best_bid_level.begin());
        }
        if (ask_order.quantity <= 0) {
            order_locations_.erase(ask_order.order_id);
            best_ask_level.erase(best_ask_level.begin());
        }

        if (best_bid_level.empty()) {
            bids_.erase(bids_.begin());
        }
        if (best_ask_level.empty()) {
            asks_.erase(asks_.begin());
        }
    }
    return trades;
}


std::pair<double, double> OrderBook::get_BBO() const {
    double best_bid = bids_.empty() ? 0.0 : bids_.begin()->first; // Or some other indicator for no bid
    double best_ask = asks_.empty() ? std::numeric_limits<double>::infinity() : asks_.begin()->first; // Or some indicator for no ask
    return {best_bid, best_ask};
}

int OrderBook::get_depth_at_price(double price, OrderSide side) const {
    // Fix: Use separate handling for bid and ask sides instead of conditional operator
    int total_quantity = 0;
    
    if (side == OrderSide::BID) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            for (const auto& order : it->second) {
                total_quantity += order.quantity;
            }
        }
    } else { // ASK
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            for (const auto& order : it->second) {
                total_quantity += order.quantity;
            }
        }
    }
    
    return total_quantity;
}

int OrderBook::get_total_volume(OrderSide side) const {
    // Fix: Use separate handling for bid and ask sides instead of conditional operator
    int total_volume = 0;
    
    if (side == OrderSide::BID) {
        for (const auto& price_level_pair : bids_) {
            for (const auto& order : price_level_pair.second) {
                total_volume += order.quantity;
            }
        }
    } else { // ASK
        for (const auto& price_level_pair : asks_) {
            for (const auto& order : price_level_pair.second) {
                total_volume += order.quantity;
            }
        }
    }
    
    return total_volume;
}

bool OrderBook::has_order(const std::string& order_id) const {
    return order_locations_.count(order_id);
}

void OrderBook::register_agent(MarketMakerAgent* agent) {
    mm_agent_ = agent;
}

void OrderBook::notify_agent_of_fill(const Order& order, double fill_price, int fill_quantity) {
    if (mm_agent_) {
        // Check if the order belongs to the registered agent.
        // This requires the agent to expose its order IDs or for the agent to check itself.
        // For now, we pass all fills to the agent, and it can filter.
        mm_agent_->on_fill(order, fill_price, fill_quantity);
    }
}

// ----------------------------------------------------------------------------
// Content from MarketMakerAgent.cpp (Method Implementations)
// ----------------------------------------------------------------------------
MarketMakerAgent::MarketMakerAgent(
    OrderBook& order_book,
    double initial_capital,
    int target_inv, // Renamed to avoid conflict with member
    int max_inv,    // Renamed to avoid conflict with member
    double base_spread_param, // Renamed
    double inventory_skew_factor_param, // Renamed
    int quote_qty // Renamed
) : order_book_(order_book),
    capital_(initial_capital), // Now this field is initialized and will be used
    inventory_(0),
    target_inventory_(target_inv),
    max_inventory_(max_inv),
    realized_pnl_(0.0),
    unrealized_pnl_(0.0),
    base_spread_(base_spread_param),
    inventory_skew_factor_(inventory_skew_factor_param),
    quote_quantity_(quote_qty),
    avg_entry_price_(0.0),
    current_bid_id_(""),
    current_ask_id_(""),
    current_bid_(0.0),
    current_ask_(0.0),
    total_trades_(0),
    total_volume_(0.0),
    sum_pnl_per_trade_(0.0),
    sum_squared_pnl_per_trade_(0.0),
    max_inventory_held_(0),
    sum_spread_captured_(0.0) {
    order_book_.register_agent(this); // Register with the order book
    record_state(); // Initial state
}

void MarketMakerAgent::update_quotes(double fair_value) {
    cancel_existing_orders();
    place_quotes(fair_value);
    update_unrealized_pnl(fair_value); // Update based on current FV
    record_state();
}

void MarketMakerAgent::on_fill(const Order& filled_order, double fill_price, int fill_quantity) {
    // Check if the filled order is one of this agent's current quotes
    bool is_my_order = (filled_order.order_id == current_bid_id_ || filled_order.order_id == current_ask_id_);

    if (is_my_order) {
        total_trades_++;
        total_volume_ += fill_quantity; // Add to total volume
        max_inventory_held_ = std::max(max_inventory_held_, std::abs(inventory_)); // Update max inventory held

        if (filled_order.order_id == current_bid_id_) { // Our bid was hit (we bought)
            // Update average entry price before updating inventory
            if (inventory_ >= 0) { // Was flat or long, adding to long or creating long
                 avg_entry_price_ = (avg_entry_price_ * inventory_ + fill_price * fill_quantity) / (inventory_ + fill_quantity);
            } else { // Was short, buying to cover. Realize PNL for covered part.
                int quantity_to_cover = std::min(fill_quantity, -inventory_);
                realized_pnl_ += (avg_entry_price_ - fill_price) * quantity_to_cover;
                sum_pnl_per_trade_ += (avg_entry_price_ - fill_price) * quantity_to_cover; // Simplified PNL for this part
                sum_squared_pnl_per_trade_ += std::pow((avg_entry_price_ - fill_price) * quantity_to_cover, 2);


                if (fill_quantity > quantity_to_cover) { // Flipped to long
                    avg_entry_price_ = fill_price; // New position starts at this price
                } else if (inventory_ + quantity_to_cover == 0) { // Became flat
                     avg_entry_price_ = 0; // No position, no avg price
                }
                // If still short but less short, avg_entry_price_ remains for the short.
            }
            inventory_ += fill_quantity;
            current_bid_id_ = ""; // Order is fully or partially filled, assume we'd replace
        } else if (filled_order.order_id == current_ask_id_) { // Our ask was hit (we sold)
            // Update average entry price before updating inventory (if going short or adding to short)
            if (inventory_ <= 0) { // Was flat or short, adding to short or creating short
                avg_entry_price_ = (avg_entry_price_ * (-inventory_) + fill_price * fill_quantity) / ((-inventory_) + fill_quantity);
            } else { // Was long, selling to reduce or go short. Realize PNL.
                int quantity_to_liquidate = std::min(fill_quantity, inventory_);
                realized_pnl_ += (fill_price - avg_entry_price_) * quantity_to_liquidate;
                sum_pnl_per_trade_ += (fill_price - avg_entry_price_) * quantity_to_liquidate;
                sum_squared_pnl_per_trade_ += std::pow((fill_price - avg_entry_price_) * quantity_to_liquidate, 2);

                if (fill_quantity > quantity_to_liquidate) { // Flipped to short
                    avg_entry_price_ = fill_price; // New short position starts at this price
                } else if (inventory_ - quantity_to_liquidate == 0) { // Became flat
                    avg_entry_price_ = 0;
                }
                 // If still long but less long, avg_entry_price_ remains for the long.
            }
            inventory_ -= fill_quantity;
            current_ask_id_ = ""; // Order is filled
        }

        // Capture spread if a buy and sell pair effectively closes a round trip near target inventory
        if ((filled_order.side == OrderSide::BID && inventory_ > target_inventory_) ||
            (filled_order.side == OrderSide::ASK && inventory_ < target_inventory_)) {
            // This logic for spread capture is simplistic. A better way is to track individual trades.
            // For now, let's assume if a fill happens and we had quotes out, we captured some spread.
            // This is a rough estimate.
            if(current_bid_ > 0 && current_ask_ > 0 && current_ask_ > current_bid_) {
                 sum_spread_captured_ += (current_ask_ - current_bid_);
            }
        }

        // Update capital based on cash flow from trade
        if (filled_order.order_id == current_bid_id_) {
            capital_ -= fill_price * fill_quantity; // Cash outflow when buying
        } else if (filled_order.order_id == current_ask_id_) {
            capital_ += fill_price * fill_quantity; // Cash inflow when selling
        }

        // It's important to update unrealized P&L after inventory and avg_entry_price_ changes.
        // The fair_value for this update should ideally be the current market mid-price.
        // We'll rely on the next call to update_quotes to pass the latest fair_value.
        // For immediate consistency after a fill, one might pass the fill_price or last known FV.
        // Here, we assume update_unrealized_pnl will be called soon with a fresh FV.
        record_state(); // Record state after fill
    }
}


void MarketMakerAgent::update_unrealized_pnl(double fair_value) {
    if (inventory_ == 0) {
        unrealized_pnl_ = 0.0;
    } else if (inventory_ > 0) { // Long position
        unrealized_pnl_ = (fair_value - avg_entry_price_) * inventory_;
    } else { // Short position (inventory_ < 0)
        unrealized_pnl_ = (avg_entry_price_ - fair_value) * (-inventory_);
    }
}

void MarketMakerAgent::get_stats(
    double& total_pnl_out, double& avg_pnl_per_trade_out,
    double& stddev_pnl_out, double& sharpe_ratio_out,
    int& max_inventory_out, int& total_trades_out,
    double& avg_spread_captured_out
) const {
    total_pnl_out = realized_pnl_ + unrealized_pnl_; // Current total P&L
    max_inventory_out = max_inventory_held_;
    total_trades_out = total_trades_;

    if (total_trades_ > 0) {
        avg_pnl_per_trade_out = sum_pnl_per_trade_ / total_trades_;
        double variance = (sum_squared_pnl_per_trade_ / total_trades_) - (avg_pnl_per_trade_out * avg_pnl_per_trade_out);
        stddev_pnl_out = std::sqrt(std::max(0.0, variance)); // Ensure non-negative for sqrt
        sharpe_ratio_out = (stddev_pnl_out > 1e-9) ? (avg_pnl_per_trade_out / stddev_pnl_out) * std::sqrt(252.0) : 0.0; // Annualized, assuming daily trades
        avg_spread_captured_out = sum_spread_captured_ / total_trades_ ; // This sum_spread_captured_ needs to be correctly updated.
    } else {
        avg_pnl_per_trade_out = 0.0;
        stddev_pnl_out = 0.0;
        sharpe_ratio_out = 0.0;
        avg_spread_captured_out = 0.0;
    }
}

void MarketMakerAgent::cancel_existing_orders() {
    if (!current_bid_id_.empty()) {
        order_book_.cancel_order(current_bid_id_);
        current_bid_id_ = "";
    }
    if (!current_ask_id_.empty()) {
        order_book_.cancel_order(current_ask_id_);
        current_ask_id_ = "";
    }
}

void MarketMakerAgent::place_quotes(double fair_value) {
    // Calculate inventory-adjusted prices
    double inventory_skew = (inventory_ - target_inventory_) * inventory_skew_factor_;
    double bid_price = fair_value - base_spread_ / 2.0 - inventory_skew;
    double ask_price = fair_value + base_spread_ / 2.0 - inventory_skew;

    // Ensure minimum spread and positive prices
    bid_price = std::max(0.01, bid_price);
    ask_price = std::max(bid_price + 0.01, ask_price);

    // Limit maximum risk by reducing quote size or skipping if above max inventory
    int adjusted_bid_quantity = quote_quantity_;
    int adjusted_ask_quantity = quote_quantity_;

    // Risk management: reduce quote size as inventory approaches limits
    if (inventory_ > max_inventory_ - quote_quantity_) {
        adjusted_bid_quantity = std::max(0, max_inventory_ - inventory_);
    } else if (inventory_ < -max_inventory_ + quote_quantity_) {
        adjusted_ask_quantity = std::max(0, max_inventory_ + inventory_);
    }

    // Place quotes if valid quantities
    if (adjusted_bid_quantity > 0) {
        current_bid_id_ = generate_order_id();
        Order bid_order(bid_price, adjusted_bid_quantity, OrderSide::BID, OrderType::LIMIT, current_bid_id_);
        current_bid_ = bid_price;
        order_book_.add_order(bid_order);
    }

    if (adjusted_ask_quantity > 0) {
        current_ask_id_ = generate_order_id();
        Order ask_order(ask_price, adjusted_ask_quantity, OrderSide::ASK, OrderType::LIMIT, current_ask_id_);
        current_ask_ = ask_price;
        order_book_.add_order(ask_order);
    }
}

std::string MarketMakerAgent::generate_order_id() const {
    // Create a unique timestamp-based order ID
    static int counter = 0;
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::stringstream ss;
    ss << "MM_" << ms << "_" << counter++;
    return ss.str();
}

void MarketMakerAgent::record_state() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    pnl_history_.emplace_back(timestamp, realized_pnl_ + unrealized_pnl_);
    inventory_history_.emplace_back(timestamp, inventory_);
    quote_history_.emplace_back(timestamp, std::make_pair(current_bid_, current_ask_));
}

// ----------------------------------------------------------------------------
// Content from MarketSimulator.cpp (Method Implementations)
// ----------------------------------------------------------------------------
MarketSimulator::MarketSimulator(
    OrderBook& order_book,
    MarketMakerAgent& agent,
    double initial_price,
    double price_volatility,
    double mean_reversion_strength,
    double order_arrival_rate,
    double market_order_probability,
    double price_impact_factor,
    int min_quantity,
    int max_quantity
) : order_book_(order_book), 
    agent_(agent), 
    current_price_(initial_price),
    price_volatility_(price_volatility),
    mean_reversion_strength_(mean_reversion_strength),
    initial_price_(initial_price),
    order_arrival_rate_(order_arrival_rate),
    market_order_probability_(market_order_probability),
    price_impact_factor_(price_impact_factor),
    min_quantity_(min_quantity),
    max_quantity_(max_quantity),
    rng_(std::random_device{}()),
    price_shock_dist_(0.0, price_volatility),
    uniform_dist_(0.0, 1.0),
    quantity_dist_(min_quantity, max_quantity) {
    
    // Initialize price history with starting price
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    price_history_.emplace_back(timestamp, current_price_);
}

void MarketSimulator::run_simulation_step() {
    // Update the price model first to get new fair value
    update_price_model();
    
    // Have the market maker update its quotes based on new fair value
    agent_.update_quotes(current_price_);
    
    // Possibly generate random participant order based on order arrival rate
    if (uniform_dist_(rng_) < order_arrival_rate_) {
        generate_random_order();
    }
    
    // Record the price after this step
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    price_history_.emplace_back(timestamp, current_price_);
}

void MarketSimulator::run_simulation(int num_steps) {
    for (int i = 0; i < num_steps; ++i) {
        run_simulation_step();
    }
}

void MarketSimulator::generate_random_order() {
    // Determine if this is a buy or sell order
    bool is_buy = uniform_dist_(rng_) < 0.5;
    OrderSide side = is_buy ? OrderSide::BID : OrderSide::ASK;
    
    // Determine if it's a market or limit order
    bool is_market = uniform_dist_(rng_) < market_order_probability_;
    OrderType type = is_market ? OrderType::MARKET : OrderType::LIMIT;
    
    // Generate a random quantity
    int quantity = quantity_dist_(rng_);
    
    // Generate a price for limit orders
    double price = 0.0;
    if (!is_market) {
        // For limit orders, price is based on current price with some random offset
        double price_offset = price_volatility_ * (uniform_dist_(rng_) * 2.0 - 1.0);
        price = current_price_ + price_offset;
        
        // For buy limit orders, usually below current price; for sell, usually above
        if (is_buy) {
            price = std::max(0.01, price - price_volatility_ * 0.5); // Buy limit orders more likely below market
        } else {
            price = std::max(0.01, price + price_volatility_ * 0.5); // Sell limit orders more likely above market
        }
    }
    
    // Create the order
    std::string order_id = generate_participant_id();
    Order order(price, quantity, side, type, order_id);
    
    // Process the order
    std::vector<Trade> trades;
    if (is_market) {
        trades = order_book_.match_market_order(order);
    } else {
        order_book_.add_order(order);
        trades = order_book_.match_limit_orders(); // Check for any matches
    }
    
    // Record trades and impact price model
    for (const auto& trade : trades) {
        trade_history_.push_back(trade);
        
        // Market impact: trades move the price
        double impact = trade.quantity * price_impact_factor_ * (is_buy ? 1.0 : -1.0);
        current_price_ += impact;
        current_price_ = std::max(0.01, current_price_); // Ensure positive price
    }
}

void MarketSimulator::update_price_model() {
    // Apply mean reversion
    double mean_reversion = mean_reversion_strength_ * (initial_price_ - current_price_);
    
    // Apply random shock
    double price_shock = price_shock_dist_(rng_);
    
    // Update price
    current_price_ += mean_reversion + price_shock;
    current_price_ = std::max(0.01, current_price_); // Ensure positive price
}

std::string MarketSimulator::generate_participant_id() const {
    static int participant_counter = 0;
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::stringstream ss;
    ss << "PART_" << ms << "_" << participant_counter++;
    return ss.str();
}

// ----------------------------------------------------------------------------
// Main function and utilities for the simulator app
// ----------------------------------------------------------------------------

// Function to save simulation results to CSV
void save_simulation_results(
    const std::string& filename,
    const std::vector<std::pair<std::int64_t, double>>& price_history,
    const std::vector<std::pair<std::int64_t, double>>& pnl_history,
    const std::vector<std::pair<std::int64_t, int>>& inventory_history,
    const std::vector<std::pair<std::int64_t, std::pair<double, double>>>& quote_history) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write CSV header
    file << "Timestamp,Price,PnL,Inventory,BidPrice,AskPrice" << std::endl;
    
    // Collect all timestamps from all sources to create a unified timeline
    std::set<std::int64_t> all_timestamps;
    
    for (const auto& entry : price_history) {
        all_timestamps.insert(entry.first);
    }
    for (const auto& entry : pnl_history) {
        all_timestamps.insert(entry.first);
    }
    for (const auto& entry : inventory_history) {
        all_timestamps.insert(entry.first);
    }
    for (const auto& entry : quote_history) {
        all_timestamps.insert(entry.first);
    }
    
    // Variables to store the most recent values
    double last_price = 0.0;
    double last_pnl = 0.0;
    int last_inventory = 0;
    double last_bid = 0.0;
    double last_ask = 0.0;
    
    // Create maps for fast lookups based on timestamp
    std::map<std::int64_t, double> price_map;
    std::map<std::int64_t, double> pnl_map;
    std::map<std::int64_t, int> inventory_map;
    std::map<std::int64_t, std::pair<double, double>> quote_map;
    
    for (const auto& entry : price_history) {
        price_map[entry.first] = entry.second;
    }
    for (const auto& entry : pnl_history) {
        pnl_map[entry.first] = entry.second;
    }
    for (const auto& entry : inventory_history) {
        inventory_map[entry.first] = entry.second;
    }
    for (const auto& entry : quote_history) {
        quote_map[entry.first] = entry.second;
    }
    
    // Write data for all timestamps
    for (std::int64_t timestamp : all_timestamps) {
        // Update values if we have new data for this timestamp
        if (price_map.count(timestamp)) {
            last_price = price_map[timestamp];
        }
        if (pnl_map.count(timestamp)) {
            last_pnl = pnl_map[timestamp];
        }
        if (inventory_map.count(timestamp)) {
            last_inventory = inventory_map[timestamp];
        }
        if (quote_map.count(timestamp)) {
            last_bid = quote_map[timestamp].first;
            last_ask = quote_map[timestamp].second;
        }
        
        // Write row to CSV
        file << timestamp << ","
             << last_price << ","
             << last_pnl << ","
             << last_inventory << ","
             << last_bid << ","
             << last_ask << std::endl;
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

// Function to print simulation statistics
void print_simulation_stats(
    const MarketMakerAgent& agent,
    const MarketSimulator& simulator,
    int num_steps) {
    
    double total_pnl, avg_pnl_per_trade, stddev_pnl, sharpe_ratio, avg_spread_captured;
    int max_inventory, total_trades;
    
    agent.get_stats(
        total_pnl, avg_pnl_per_trade, stddev_pnl, sharpe_ratio,
        max_inventory, total_trades, avg_spread_captured
    );
    
    std::cout << "===== Simulation Results =====" << std::endl;
    std::cout << "Number of steps: " << num_steps << std::endl;
    std::cout << "Final price: " << simulator.get_current_price() << std::endl;
    std::cout << "Total P&L: " << total_pnl << std::endl;
    std::cout << "Realized P&L: " << agent.get_realized_pnl() << std::endl;
    std::cout << "Unrealized P&L: " << agent.get_unrealized_pnl() << std::endl;
    std::cout << "Current inventory: " << agent.get_inventory() << std::endl;
    std::cout << "Max inventory held: " << max_inventory << std::endl;
    std::cout << "Total trades: " << total_trades << std::endl;
    std::cout << "Average P&L per trade: " << avg_pnl_per_trade << std::endl;
    std::cout << "P&L standard deviation: " << stddev_pnl << std::endl;
    std::cout << "Sharpe ratio (annualized): " << sharpe_ratio << std::endl;
    std::cout << "Average spread captured: " << avg_spread_captured << std::endl;
    std::cout << "============================" << std::endl;
}

// Function to parse simulation parameters from config file or command line
struct SimulationConfig {
    double initial_price = 100.0;
    double price_volatility = 0.5;
    double mean_reversion_strength = 0.1;
    int num_steps = 1000;
    double order_arrival_rate = 0.5;
    double market_order_probability = 0.2;
    double price_impact_factor = 0.01;
    int min_quantity = 1;
    int max_quantity = 10;
    double initial_capital = 10000000.0;
    int target_inventory = 0;
    int max_inventory = 100;
    double base_spread = 1.0;
    double inventory_skew_factor = 0.1;
    int quote_quantity = 10;
    std::string output_file = "simulation_results.csv";
    
    bool parse_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening config file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::istringstream iss(line);
            std::string key;
            if (std::getline(iss, key, '=')) {
                std::string value;
                if (std::getline(iss, value)) {
                    // Trim whitespace
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    
                    if (key == "initial_price") {
                        initial_price = std::stod(value);
                    } else if (key == "price_volatility") {
                        price_volatility = std::stod(value);
                    } else if (key == "mean_reversion_strength") {
                        mean_reversion_strength = std::stod(value);
                    } else if (key == "num_steps") {
                        num_steps = std::stoi(value);
                    } else if (key == "order_arrival_rate") {
                        order_arrival_rate = std::stod(value);
                    } else if (key == "market_order_probability") {
                        market_order_probability = std::stod(value);
                    } else if (key == "price_impact_factor") {
                        price_impact_factor = std::stod(value);
                    } else if (key == "min_quantity") {
                        min_quantity = std::stoi(value);
                    } else if (key == "max_quantity") {
                        max_quantity = std::stoi(value);
                    } else if (key == "initial_capital") {
                        initial_capital = std::stod(value);
                    } else if (key == "target_inventory") {
                        target_inventory = std::stoi(value);
                    } else if (key == "max_inventory") {
                        max_inventory = std::stoi(value);
                    } else if (key == "base_spread") {
                        base_spread = std::stod(value);
                    } else if (key == "inventory_skew_factor") {
                        inventory_skew_factor = std::stod(value);
                    } else if (key == "quote_quantity") {
                        quote_quantity = std::stoi(value);
                    } else if (key == "output_file") {
                        output_file = value;
                    } else {
                        std::cerr << "Unknown parameter: " << key << std::endl;
                    }
                }
            }
        }
        
        return true;
    }
    
    void print() {
        std::cout << "===== Simulation Configuration =====" << std::endl;
        std::cout << "Initial price: " << initial_price << std::endl;
        std::cout << "Price volatility: " << price_volatility << std::endl;
        std::cout << "Mean reversion strength: " << mean_reversion_strength << std::endl;
        std::cout << "Number of steps: " << num_steps << std::endl;
        std::cout << "Order arrival rate: " << order_arrival_rate << std::endl;
        std::cout << "Market order probability: " << market_order_probability << std::endl;
        std::cout << "Price impact factor: " << price_impact_factor << std::endl;
        std::cout << "Min quantity: " << min_quantity << std::endl;
        std::cout << "Max quantity: " << max_quantity << std::endl;
        std::cout << "Initial capital: " << initial_capital << std::endl;
        std::cout << "Target inventory: " << target_inventory << std::endl;
        std::cout << "Max inventory: " << max_inventory << std::endl;
        std::cout << "Base spread: " << base_spread << std::endl;
        std::cout << "Inventory skew factor: " << inventory_skew_factor << std::endl;
        std::cout << "Quote quantity: " << quote_quantity << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "=================================" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Parse configuration
    SimulationConfig config;
    
    if (argc > 1) {
        // Check if help is requested
        std::string arg1 = argv[1];
        if (arg1 == "-h" || arg1 == "--help") {
            std::cout << "Usage: " << argv[0] << " [config_file]" << std::endl;
            std::cout << "If no config file is provided, default parameters will be used." << std::endl;
            return 0;
        }
        
        // Otherwise, treat first argument as config file
        if (!config.parse_from_file(argv[1])) {
            std::cerr << "Using default parameters." << std::endl;
        }
    } else {
        std::cout << "No config file provided. Using default parameters." << std::endl;
    }
    
    // Print configuration
    config.print();
    
    // Initialize components
    OrderBook order_book;
    
    MarketMakerAgent agent(
        order_book,
        config.initial_capital,
        config.target_inventory,
        config.max_inventory,
        config.base_spread,
        config.inventory_skew_factor,
        config.quote_quantity
    );
    
    MarketSimulator simulator(
        order_book,
        agent,
        config.initial_price,
        config.price_volatility,
        config.mean_reversion_strength,
        config.order_arrival_rate,
        config.market_order_probability,
        config.price_impact_factor,
        config.min_quantity,
        config.max_quantity
    );
    
    // Run simulation
    std::cout << "Running simulation..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    simulator.run_simulation(config.num_steps);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Simulation completed in " << duration << " ms" << std::endl;
    
    // Print statistics
    print_simulation_stats(agent, simulator, config.num_steps);
    
    // Save results
    save_simulation_results(
        config.output_file,
        simulator.get_price_history(),
        agent.get_pnl_history(),
        agent.get_inventory_history(),
        agent.get_quote_history()
    );
    
    return 0;
}
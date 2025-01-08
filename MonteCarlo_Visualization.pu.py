import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MonteCarloVisualizer:
  def __init__(self, stock_prices_file: str, payoffs_file: str):
    self.stock_prices_file = stock_prices_file
    self.payoffs_file = payoffs_file
    self.stock_prices = None
    self.payoffs = None

  def load_data(self):
    self.stock_prices = pd.read_csv(self.stock_prices_file, header=None, names=["Stock Price"])
    self.payoffs = pd.read_csv(self.payoffs_file, header=None, names=["Payoff"])
    print(f"Data loaded:\n- Stock prices: {len(self.stock_prices)} entries\n- Payoffs: {len(self.payoffs)} entries")
  
  def visualize_stock_prices(self):
    if self.stock_prices is not None:
      self.plot_histogram(
        self.stock_prices["Stock Price"],
        title="Histogram of Simulated Stock Prices",
        xlabel="Stock Price",
        ylabel="Frequency",
        color="skyblue"
      )
    else:
      print("Stock prices data not loaded.")
  def visualize_convergence(self):
    if self.stock_prices is not None:
      option_prices = self.stock_prices["Stock Price"].expanding().mean()
      plt.figure(figsize=(10, 6))
      plt.plot(option_prices, color="green")
      plt.title("Convergence of Option Price")
      plt.xlabel("Simulation Step")
      plt.ylabel("Average Option Price")
      plt.grid(True)
      plt.show()
    else:
      print("Stock prices data not loaded.")

if __name__ == "__main__":
  visualizer = MonteCarloVisualizer("stock_prices.csv", "payoffs.csv")
  visualizer.load_data()
  visualizer.visualize_stock_prices()
  visualizer.visualize_payoffs()
  visualizer.visualize_convergence()


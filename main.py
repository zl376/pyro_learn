import torch
import matplotlib.pyplot as plt
from src.model import linear_regression_model, generate_data
from src.inference import run_inference

def main():
    # Generate synthetic data
    x, y = generate_data()

    # Run inference
    svi = run_inference(linear_regression_model, x, y)

    # Extract the posterior parameters
    alpha_mean = pyro.param("AutoDiagonalNormal.locs.alpha").item()
    beta_mean = pyro.param("AutoDiagonalNormal.locs.beta").item()
    sigma_mean = pyro.param("AutoDiagonalNormal.locs.sigma").item()

    print(f"Inferred parameters: alpha={alpha_mean:.2f}, beta={beta_mean:.2f}, sigma={sigma_mean:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, alpha_mean + beta_mean * x, color='red', label="Inferred Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Bayesian Linear Regression with Pyro")
    plt.show()

if __name__ == "__main__":
    main()

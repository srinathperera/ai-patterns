import numpy as np
from scipy.optimize import minimize

def constrained_linear_solve(O_sim, C, N):
    """
    Finds A_sim such that C * A_sim = O_sim
    Constraints: 
    - sum(A_sim) = N
    - A_sim[i] >= 0
    """
    # Number of variables
    n = C.shape[1]
    
    # Objective function: Minimize the L2 norm of (C * A_sim - O_sim)
    # We want the residual to be as close to zero as possible
    def objective(A):
        residual = np.dot(C, A) - O_sim
        return np.sum(residual**2)

    # Constraint: sum(A_sim) - N = 0
    cons = ({'type': 'eq', 'fun': lambda A: np.sum(A) - N})
    
    # Bounds: Each element in A_sim must be >= 0
    # (Using 0 as a practical bound for "greater than zero")
    bounds = [(0, None) for _ in range(n)]
    
    # Initial guess: Distribute N equally across all elements
    initial_guess = np.ones(n) * (N / n)
    
    # Perform optimization using SLSQP (Sequential Least Squares Programming)
    res = minimize(objective, initial_guess, method='SLSQP', 
                   bounds=bounds, constraints=cons)
    
    if res.success:
        return res.x
    else:
        raise ValueError(f"Optimization failed: {res.message}")


def constrained_linear_solve_xx(C, observed_proportions):
    """
    Solves for p in the equation C * p = observed_proportions
    subject to:
        1. sum(p) = 1
        2. 0 <= p_i <= 1 for all i
    """
    n = C.shape[1]
    
    # Objective function: Minimize the squared error ||C*p - obs||^2
    def objective(p):
        return np.sum((np.dot(C, p) - observed_proportions)**2)
    
    # Constraint: sum(p) must be 1
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})
    
    # Bounds: each p_i must be between 0 and 1
    bounds = [(0, 1) for _ in range(n)]
    
    # Initial guess: uniform distribution
    p0 = np.ones(n) / n
    
    res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res.success:
        return res.x
    else:
        raise ValueError("Optimization failed: " + res.message)

# Example Usage:
# C = np.array([[0.9, 0.1], [0.1, 0.9]]) # High accuracy model
# obs = np.array([0.4, 0.6])            # Observed proportions
# print(constrained_linear_solve(C, obs))

def dig1(C_norm):
    # 2. Get the INVERSE matrix directly
    C_inv = np.linalg.inv(C_norm)

    # 3. Look at Row 4 of the INVERSE matrix
    # These are the weights applied to your observed counts O to get Actual_4
    weights_for_class_4 = C_inv[4]

    print("Weights applied to each observed class count to estimate Class 4:")
    for i, w in enumerate(weights_for_class_4):
        print(f"Class {i}: {w:.4f}")

def estimate_with_bounds(conf_matrix, observed_counts, n_iterations=2000):
    C = np.array(conf_matrix, dtype=float)
    O = np.array(observed_counts, dtype=float)
    n_samples = int(O.sum())
    n_classes = len(O)
    
    # Normalize the matrix once
    C_norm = C / (C.sum(axis=0) + 1e-12)
    
    results = []
    
    # Run simulations
    for _ in range(n_iterations):
        # 1. Simulate sampling noise in the 5000 predictions
        # We draw a new 'O' based on the observed distribution
        O_sim = np.random.multinomial(n_samples, O / n_samples)
        
        try:
            # 2. Solve for this iteration
            A_sim = np.linalg.solve(C_norm, O_sim)
            #A_sim = constrained_linear_solve(O_sim, C_norm, n_samples)
            
            #print("observed_counts[5], O_sim[5], A_sim[5]:", observed_counts[4], O_sim[4], A_sim[4])
            A_sim = np.clip(A_sim, 0, None)
            A_sim = (A_sim / A_sim.sum()) * n_samples
            
            results.append(A_sim)
        except np.linalg.LinAlgError:
            continue

    results = np.array(results)
    
    # 3. Calculate Percentiles (90% bounds use 5th and 95th)
    lower_bound = np.percentile(results, 5, axis=0)
    median_est = np.percentile(results, 50, axis=0)
    upper_bound = np.percentile(results, 95, axis=0)

    
    
    return lower_bound, median_est, upper_bound


np.random.seed(42)
# Create a 10x10 matrix with high diagonal values (good model)
#random_cm = np.random.rand(8, 8) * 10
#np.fill_diagonal(random_cm, 80) # Strong diagonal performance

#random_cm = np.array([
#    33, 0, 2, 0, 1, 4, 2, 0,
#    1, 3, 0, 1, 2, 2, 0, 1,
#    4, 0, 15, 0, 0, 2, 0, 0,
#    1, 0, 0, 6, 3, 1, 1, 1,
#    0, 3, 0, 0, 9, 0, 0, 5,
#    4, 0, 5, 1, 0, 16, 1, 0,
#    0, 0, 0, 1, 2, 3, 11, 0,
#    0, 0, 1, 2, 3, 1, 1, 10
#]).reshape(8, 8)

#random_cm = np.array([
#   33, 0, 2, 0, 1, 4, 2, 0,
#    1, 3, 0, 1, 2, 2, 0, 1,
#    4, 0, 15, 0, 0, 2, 0, 0,
#    1, 0, 0, 6, 3, 1, 1, 1,
#    0, 3, 0, 0, 9, 0, 0, 5,
#    4, 0, 5, 1, 0, 16, 1, 0,
#    0, 0, 0, 1, 2, 3, 11, 0,
#    0, 0, 1, 2, 3, 1, 1, 10
#]).reshape(8, 8)


random_cm = np.array([
    [6, 1, 0, 4, 2, 1, 4, 2],   # Agent Architecture 60
    [0, 50, 6, 0, 2, 0, 1, 4],  # Forecasting with Classical Models 259
    [0, 6, 38, 0, 3, 0, 0, 5],  # LLM-based Multimodal Generative Prompting 246
    [2, 1, 0, 10, 1, 1, 3, 2],  # Model Abstraction 34
    [0, 4, 8, 0, 25, 3, 1, 9],  # Preprocessing Text and Numerical Data 166
    [0, 0, 0, 1, 4, 18, 2, 3],  # Retrieval Augmented Generation(RAG) 71
    [2, 0, 1, 5, 1, 0, 11, 11], # Using Tools with LLMs 61
    [2, 10, 8, 2, 9, 4, 4, 46]  # none 369
]).reshape(8, 8)




#print confusion matrix formatted as a table to show two decimal places
print("Confusion Matrix:")
print(np.round(random_cm, 2).tolist())
print("-" * 55)
# Print each row as a formatted list
for i in range(len(random_cm)):
    row_str = " ".join([f"{val:6.2f}" for val in random_cm[i]])
    print(f"Row {i:<3} | {row_str}")
print("-" * 55)

# Let's say your model predicted these counts for the 10 classes (sums to 5000)
#observed_counts_10 = np.random.multinomial(5000, [0.1]*10)
#observed_counts_10 = [275, 88, 171, 52, 70, 219, 71, 180]
observed_counts_10 = [60, 259, 246, 34, 166, 71, 61, 369]
#print observed counts
print("Observed Counts:")
print(observed_counts_10)
# --- Example for 10 classes ---
# Using the random_cm and observed_counts_10 from previous step
low, med, high = estimate_with_bounds(random_cm, observed_counts_10, n_iterations=10000)

print(f"{'Class':<6} | {'Lower (5%)':<12} | {'Median':<12} | {'Upper (95%)':<12}")
print("-" * 55)
for i in range(len(med)):
    print(f"{i:<6} | {low[i]:<12.1f} | {med[i]:<12.1f} | {high[i]:<12.1f}")
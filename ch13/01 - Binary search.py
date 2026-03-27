import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import math
    import random
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    return math, np, plt, random


@app.cell
def _(math, random):
    def entropy(num_states):
        if num_states <= 0:
            return 0
        return math.log2(num_states)

    def binary_search_with_entropy(target, low=1, high=100):
        steps = 0
        entropies = []
        while low <= high:
            steps = steps + 1
            num_states = high - low + 1
            entropies.append(entropy(num_states))
            mid = (low + high) // 2  # Track entropy before the guess
            if mid == target:
                break
            elif target > mid:
                low = mid + 1
            else:
                high = mid - 1
        entropies.append(0)
        return (steps, entropies)

    def random_guess_with_entropy(target, low=1, high=100):
        steps = 0
        possible = list(range(low, high + 1))
        entropies = []  # Final entropy (should be 0)
        while possible:
            steps = steps + 1
            entropies.append(entropy(len(possible)))
            guess = random.choice(possible)
            if guess == target:
                break
            if guess < target:
                possible = [x for x in possible if x > guess]
            else:
                possible = [x for x in possible if x < guess]
        entropies.append(0)
        return (steps, entropies)  # Track entropy before guess  # Eliminate inconsistent values  # Final entropy

    return binary_search_with_entropy, random_guess_with_entropy


@app.cell
def _(binary_search_with_entropy, plt, random, random_guess_with_entropy):
    random.seed(0)

    target = 73

    bs_steps, bs_entropy = binary_search_with_entropy(target)
    rg_steps, rg_entropy = random_guess_with_entropy(target)

    plt.figure()

    plt.plot(bs_entropy, marker='o', label='Binary Search')
    plt.plot(rg_entropy, marker='o', label='Random Guessing')

    plt.xlabel("Step")
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy Reduction: Binary Search vs Random Guessing")
    plt.legend()

    plt.show()
    return


@app.cell
def _(binary_search_with_entropy, random, random_guess_with_entropy):
    def run_experiments(n_trials=1000, low=1, high=100):
        bs_steps_all = []
        rg_steps_all = []
    
        bs_entropy_curves = []
        rg_entropy_curves = []
    
        max_len = 0
    
        for _ in range(n_trials):
            target = random.randint(low, high)
        
            bs_steps, bs_e = binary_search_with_entropy(target, low, high)
            rg_steps, rg_e = random_guess_with_entropy(target, low, high)
        
            bs_steps_all.append(bs_steps)
            rg_steps_all.append(rg_steps)
        
            bs_entropy_curves.append(bs_e)
            rg_entropy_curves.append(rg_e)
        
            max_len = max(max_len, len(bs_e), len(rg_e))
    
        return {
            "bs_steps": bs_steps_all,
            "rg_steps": rg_steps_all,
            "bs_entropy": bs_entropy_curves,
            "rg_entropy": rg_entropy_curves,
            "max_len": max_len
        }

    return (run_experiments,)


@app.function
def average_curves(curves, max_len):
    padded = []
    
    for c in curves:
        padded.append(c + [0] * (max_len - len(c)))
    
    avg = [sum(step) / len(step) for step in zip(*padded)]
    return avg


@app.cell
def _(plt, run_experiments):
    results = run_experiments(1000)

    bs_avg = average_curves(results["bs_entropy"], results["max_len"])
    rg_avg = average_curves(results["rg_entropy"], results["max_len"])

    plt.figure()

    plt.plot(bs_avg, label="Binary Search")
    plt.plot(rg_avg, linestyle="--", label="Random Guessing")

    plt.xlabel("Step")
    plt.ylabel("Entropy (bits)")
    plt.title("Average Entropy Reduction Across Trials")
    plt.legend()

    plt.show()
    return (results,)


@app.cell
def _(np, results):
    bs_steps_1 = np.array(results['bs_steps'])
    rg_steps_1 = np.array(results['rg_steps'])
    print('Binary Search:')
    print(f'  Mean steps: {bs_steps_1.mean():.2f}')
    print(f'  Std:        {bs_steps_1.std():.2f}')
    print('\nRandom Guessing:')
    print(f'  Mean steps: {rg_steps_1.mean():.2f}')
    print(f'  Std:        {rg_steps_1.std():.2f}')
    return bs_steps_1, rg_steps_1


@app.cell
def _(bs_steps_1, plt, rg_steps_1):
    plt.hist(bs_steps_1, bins=range(1, max(bs_steps_1) + 2), alpha=0.6, label='Binary Search')
    plt.hist(rg_steps_1, bins=range(1, max(rg_steps_1) + 2), alpha=0.6, hatch='//', label='Random Guessing')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.title('Distribution of Search Efficiency')
    plt.legend()
    plt.show()
    return


@app.cell
def _(bs_steps_1, rg_steps_1):
    efficiency = rg_steps_1 / bs_steps_1
    print(f'Average efficiency gap: {efficiency.mean():.2f}x slower')
    return


if __name__ == "__main__":
    app.run()

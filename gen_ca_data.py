import json
import random

# Single-dimensional cellular automaton (Wolfram code)
def rule_to_dict(rule_number):
    """Convert rule number (0-255) to a mapping from 3-bit neighborhood to output."""
    rule_bin = f"{rule_number:08b}"  # Reverse for Wolfram convention
    keys = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
    ]
    return {k: int(v) for k, v in zip(keys, rule_bin)}

def next_gen(cells, rule_map):
    """Compute next generation of cells."""
    n = len(cells)
    new_cells = []
    for i in range(n):
        left = cells[i-1] if i > 0 else 0
        center = cells[i]
        right = cells[i+1] if i < n-1 else 0
        new_cells.append(rule_map[(left, center, right)])
    return new_cells

def print_cells(cells):
    print(''.join(['#' if c else ' ' for c in cells]))

def cells_to_int(cells):
    """Convert binary cell array to base-10 integer."""
    return int(''.join(str(c) for c in cells), 2)

def run_automaton(rule_number, size=32, steps=15, seed=None, verbose=False):
    """Run a single automaton simulation and return the data (does not save to file)."""
    rule_map = rule_to_dict(rule_number)
    if verbose:
        print("Neighborhoods and outputs:")
        for k in sorted(rule_map.keys(), reverse=True):
            neighborhood = ''.join(str(x) for x in k)
            print(f"  {neighborhood} -> {rule_map[k]}")
    if seed is None:
        cells = [0]*size
        cells[size//2] = 1  # Single center cell
    else:
        cells = seed[:size] + [0]*(size-len(seed))
    output = [cells.copy()]
    for _ in range(steps):
        if verbose:
            print_cells(cells)
        cells = next_gen(cells, rule_map)
        output.append(cells.copy())
    # Convert binary representations to integers
    output_integers = [cells_to_int(gen) for gen in output]
    # Return the data for this run
    data = {
        "rule": rule_number,
        "size": size,
        "steps": steps,
        "seed": seed,
        "output": output,
        "output_as_integers": output_integers
    }
    return data

def generate_rule_data(rule_number, size=32, steps=15, num_runs=10):
    """Generate multiple runs for a rule with different initial conditions and save to file."""
    print(f"Generating data for Rule {rule_number}...")
    runs = []
    
    # Run 1: Single center cell (default)
    runs.append(run_automaton(rule_number, size=size, steps=steps, seed=None))
    
    # Additional runs with random seeds
    for i in range(num_runs - 1):
        # Generate random initial conditions
        random_seed = [random.randint(0, 1) for _ in range(size)]
        runs.append(run_automaton(rule_number, size=size, steps=steps, seed=random_seed))
    
    # Save all runs to file in GENERATED_DATA folder
    filename = f"GENERATED_DATA/rule_{rule_number}.json"
    with open(filename, "w") as f:
        json.dump(runs, f, indent=2)
    
    print(f"  Saved {len(runs)} runs to {filename}")

if __name__ == "__main__":
    # Example usage: Rule 30, 31 cells, 15 steps
    # import sys
    # rule = 30
    # if len(sys.argv) > 1:
    #     rule = int(sys.argv[1])
    # generate_rule_data(rule_number=rule)
    
    for rule in range(100, 140):
        generate_rule_data(rule_number=rule, size=32, steps=100, num_runs=100)
    # generate_rule_data(rule_number=110, size=32, steps=200, num_runs=1000)

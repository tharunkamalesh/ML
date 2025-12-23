import csv

# Load CSV data
def load_data(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    return data[1:]  # skip header

# Check if hypothesis is consistent with example
def is_consistent(hypothesis, example):
    for h, e in zip(hypothesis, example):
        if h != '?' and h != e:
            return False
    return True

# Candidate-Elimination Algorithm
def candidate_elimination(data):
    num_attributes = len(data[0]) - 1

    # Initialize S and G
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]

    for example in data:
        attributes = example[:-1]
        label = example[-1]

        if label == 'Yes':
            # Remove inconsistent hypotheses from G
            G = [g for g in G if is_consistent(g, attributes)]

            # Generalize S
            for i in range(num_attributes):
                if S[i] == '0':
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = '?'

        else:  # Negative example
            new_G = []
            for g in G:
                if is_consistent(g, attributes):
                    for i in range(num_attributes):
                        if g[i] == '?':
                            if S[i] != '?' and S[i] != '0':
                                new_hypothesis = g.copy()
                                new_hypothesis[i] = S[i]
                                new_G.append(new_hypothesis)
                else:
                    new_G.append(g)
            G = new_G

    return S, G

# Run the algorithm
data = load_data("enjoysport.csv")
S, G = candidate_elimination(data)

print("Final Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")
for g in G:
    print(g)

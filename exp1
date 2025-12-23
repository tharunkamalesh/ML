# FIND-S Algorithm Implementation

# Training data
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Step 1: Initialize the most specific hypothesis
num_attributes = len(training_data[0]) - 1
hypothesis = ['0'] * num_attributes

print("Initial Hypothesis:", hypothesis)

# Step 2: Apply FIND-S
for example in training_data:
    if example[-1] == 'Yes':  # Only positive examples
        for i in range(num_attributes):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'

        print("Updated Hypothesis:", hypothesis)

# Step 3: Final hypothesis
print("\nFinal Most Specific Hypothesis:")
print(hypothesis)

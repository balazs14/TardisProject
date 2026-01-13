import json
from collections import defaultdict

# Read the original file
with open('available_options.json', 'r') as f:
    data = json.load(f)

# Group options by exchange temporarily to sample them
grouped = defaultdict(list)
for option in data:
    exchange = option.get('exchange')
    if exchange and len(grouped[exchange]) < 3:
        grouped[exchange].append(option)

# Flatten back to a list (same structure as original)
sample_data = []
for exchange_options in grouped.values():
    sample_data.extend(exchange_options)

# Save to sample_available_options.json
with open('sample_available_options.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print(f"Sample file created: sample_available_options.json")
print(f"Number of options in final JSON: {len(sample_data)}")
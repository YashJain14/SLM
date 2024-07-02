# import jsonlines

# # Input and output file paths
# input_file = "data.jsonl"
# output_file = "train.jsonl"


# # Function to reformat the structure of each entry
# def reformat_entry(entry):
#     messages = entry["messages"]
#     text = "".join(f'<|start_of_turn|>{message["role"]}\n{message["content"]}<|end_of_turn|>\n' for message in messages)
#     text =text.strip("\n")
#     return {"text": text}

# # Read from the input file, reformat, and write to the output file
# with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
#     for entry in reader:
#         reformatted_entry = reformat_entry(entry)
#         writer.write(reformatted_entry)

# print(f"Reformatted output saved to {output_file}")
import json
import random

# Function to transform each message entry, excluding system messages
def transform_entry(entry):
    transformed_text = "<bos>"
    for message in entry['messages']:
        if message['role'] == "system":
            continue
        role = "user" if message['role'] == "user" else "model"
        content = message['content']
        transformed_text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    return {"text": transformed_text.strip()}

# Read the input JSONL file
input_file = 'alldata.jsonl'
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

# Transform the data
transformed_data = [transform_entry(entry) for entry in data]

# Split the data into train, test, and validation sets
random.shuffle(transformed_data)
total = len(transformed_data)
train_end = int(0.8 * total)
val_end = train_end + int(0.1 * total)

train_data = transformed_data[:train_end]
val_data = transformed_data[train_end:val_end]
test_data = transformed_data[val_end:]

# Write the transformed and split data to JSONL files
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

write_jsonl('train.jsonl', train_data)
write_jsonl('val.jsonl', val_data)
write_jsonl('test.jsonl', test_data)

print("Data transformation and splitting completed.")

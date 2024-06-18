import jsonlines

# Input and output file paths
input_file = "data.jsonl"
output_file = "train.jsonl"


# Function to reformat the structure of each entry
def reformat_entry(entry):
    messages = entry["messages"]
    text = "".join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n' for message in messages)
    text =text.strip("\n")
    return {"text": text}

# Read from the input file, reformat, and write to the output file
with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
    for entry in reader:
        reformatted_entry = reformat_entry(entry)
        writer.write(reformatted_entry)

print(f"Reformatted output saved to {output_file}")


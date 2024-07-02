# List of file paths
jsonl_files = [
    'stackoverflowgit.jsonl',
    'stackoverflow_qa_github.jsonl',
    'stackoverflow_qa_gitlab.jsonl',
    'data.jsonl'

]

# Name of the output combined file
combined_file = 'alldata.jsonl'

# Open the output file in write mode
with open(combined_file, 'w') as outfile:
    # Iterate through each JSONL file in the list
    for filename in jsonl_files:
        # Open each file in read mode
        with open(filename, 'r') as infile:
            # Write each line from the current file to the output file
            for line in infile:
                outfile.write(line)

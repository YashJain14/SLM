import json
from collections import OrderedDict

def remove_duplicates_from_jsonl(input_file, output_file):
    unique_entries = OrderedDict()
    duplicates_removed = 0
    total_entries = 0

    # Read the input file and keep only unique entries
    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, 1):
            try:
                entry = json.loads(line)
                # Use the title (first line of user content) as the identifier
                identifier = entry['messages'][1]['content'].split('\n')[0]
                
                if identifier not in unique_entries:
                    unique_entries[identifier] = line
                else:
                    duplicates_removed += 1
                
                total_entries += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_number}")
            except KeyError:
                print(f"Unexpected structure in entry on line {line_number}")

    # Write unique entries to the output file
    with open(output_file, 'w') as outfile:
        for line in unique_entries.values():
            outfile.write(line)

    return total_entries, duplicates_removed

# Usage
input_file = 'stackoverflow_qa.jsonl'  # Replace with your input JSONL file path
output_file = 'stackoverflow_qa.jsonl'  # The new file with duplicates removed

total_entries, duplicates_removed = remove_duplicates_from_jsonl(input_file, output_file)

print(f"Total entries processed: {total_entries}")
print(f"Duplicates removed: {duplicates_removed}")
print(f"Unique entries: {total_entries - duplicates_removed}")
print(f"Unique entries have been saved to: {output_file}")
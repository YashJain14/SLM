import jsonlines  # For reading JSONL files
import tiktoken
# Assuming your num_tokens_from_string function is defined as mentioned
def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to count tokens in the JSONL file
def count_tokens_in_jsonl_file(file_path: str, role_filter: str = "assistant") -> dict:
    token_counts = {}

    # Open the JSONL file
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            # Check if the object has 'messages' field
            if 'messages' in obj and isinstance(obj['messages'], list):
                for message in obj['messages']:
                    if message.get('role') == role_filter:
                        content = message.get('content', '')
                        num_tokens = num_tokens_from_string(content)
                        
                        # Increment token count for this occurrence
                        if content in token_counts:
                            token_counts[content] += num_tokens
                        else:
                            token_counts[content] = num_tokens

    return token_counts

# Example usage:
file_path = 'data.jsonl'
token_counts = count_tokens_in_jsonl_file(file_path)

# Print token counts for each message content
counter = 0
for content, count in token_counts.items():
    print(f" Tokens Count: {count}")
    counter = counter + count
    print(counter)

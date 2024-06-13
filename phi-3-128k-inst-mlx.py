import time
import psutil
from mlx_lm import load, generate

# Function to measure time and RAM usage
def measure_performance(prompt, model_name, tokenizer_config, max_tokens):
    # Measure start time
    start_time = time.time()

    # Measure initial RAM usage
    process = psutil.Process()
    initial_ram_usage = process.memory_info().rss  # in bytes

    # Load the model and tokenizer
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)

    # Generate text
    response = generate(model,
                        tokenizer,
                        prompt=prompt,
                        verbose=False,
                        max_tokens=max_tokens)

    # Measure end time
    end_time = time.time()

    # Measure final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate time and RAM usage
    time_taken = end_time - start_time
    ram_used = final_ram_usage - initial_ram_usage

    print(f"Max tokens: {max_tokens}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"RAM used: {ram_used / (1024 ** 2):.2f} MB")
    # Print response (commented out to avoid long outputs in console)
    # print(f"Generated response: {response}")
    print("=" * 50)

# Example usage
model_name = 'mlx-community/Phi-3-mini-128k-instruct-4bit'
tokenizer_config = {"eos_token": ""}
prompt = "Tell me about openAI"
max_tokens_list = [50, 100, 200, 250, 300, 400, 500]

for max_tokens in max_tokens_list:
    measure_performance(prompt, model_name, tokenizer_config, max_tokens)


'''
MacBook Air M2 16 GB

Max tokens: 50
Time taken: 2.88 seconds
RAM used: 2118.88 MB
==================================================
Max tokens: 100
Time taken: 3.75 seconds
RAM used: 74.97 MB
==================================================
Max tokens: 200
Time taken: 6.55 seconds
RAM used: 43.61 MB
==================================================
Max tokens: 250
Time taken: 7.94 seconds
RAM used: 20.02 MB
==================================================
Max tokens: 300
Time taken: 9.43 seconds
RAM used: 16.09 MB
==================================================
Max tokens: 400
Time taken: 12.21 seconds
RAM used: 13.02 MB
==================================================
Max tokens: 500
Time taken: 15.13 seconds
RAM used: 3.33 MB
==================================================
'''
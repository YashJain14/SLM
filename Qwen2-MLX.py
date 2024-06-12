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
                        top_p=0.8,
                        temp=0.7,
                        repetition_penalty=1.05,
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
model_name = 'Qwen/Qwen2-0.5B-Instruct-MLX'
tokenizer_config = {"eos_token": ""}
prompt = "Tell me about openAI"
max_tokens_list = [50, 100, 200, 250, 300, 400, 500]

for max_tokens in max_tokens_list:
    measure_performance(prompt, model_name, tokenizer_config, max_tokens)


'''
Max tokens: 50
Time taken: 1.39 seconds
RAM used: 405.78 MB
==================================================
Max tokens: 100
Time taken: 1.63 seconds
RAM used: 52.73 MB
==================================================
Max tokens: 200
Time taken: 2.68 seconds
RAM used: 49.06 MB
==================================================
Max tokens: 250
Time taken: 3.09 seconds
RAM used: 29.23 MB
==================================================
Max tokens: 300
Time taken: 3.49 seconds
RAM used: 22.44 MB
==================================================
Max tokens: 400
Time taken: 4.47 seconds
RAM used: 22.17 MB
==================================================
Max tokens: 500
Time taken: 5.50 seconds
RAM used: 20.66 MB
==================================================
'''
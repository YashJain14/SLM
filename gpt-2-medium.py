import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to measure time and RAM usage
def measure_performance(max_length):
    # Measure start time
    start_time = time.time()

    # Measure initial RAM usage
    process = psutil.Process()
    initial_ram_usage = process.memory_info().rss  # in bytes

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    prompt = "Tell me about India."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.5,
        max_length=max_length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    # Measure end time
    end_time = time.time()

    # Measure final RAM usage
    final_ram_usage = process.memory_info().rss  # in bytes

    # Calculate time and RAM usage
    time_taken = end_time - start_time
    ram_used = final_ram_usage - initial_ram_usage

    print(f"Max length: {max_length}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"RAM used: {ram_used / (1024 ** 2):.2f} MB")
    # print(f"Generated text: {gen_text}")  # Print only the first 500 characters
    print("="*50)

# Test with different max_length values
max_lengths = [50, 100, 200, 250, 300, 400, 500]
for max_length in max_lengths:
    measure_performance(max_length)


'''
MacBook Air M2 16 GB

Max length: 50
Time taken: 117.08 seconds
RAM used: 1536.81 MB
==================================================
Max length: 100
Time taken: 6.08 seconds
RAM used: 1110.04 MB
==================================================
Max length: 200
Time taken: 11.02 seconds
RAM used: 1112.80 MB
==================================================
Max length: 250
Time taken: 13.62 seconds
RAM used: 1056.88 MB
==================================================
Max length: 300
Time taken: 15.94 seconds
RAM used: 1109.35 MB
==================================================
Max length: 400
Time taken: 21.14 seconds
RAM used: 1112.18 MB
==================================================
Max length: 500
Time taken: 26.47 seconds
RAM used: 1097.72 MB
==================================================
'''
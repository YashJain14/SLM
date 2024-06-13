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
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

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
    # print(f"Generated text: {gen_text}") 
    print("="*50)

# Test with different max_length values
max_lengths = [50, 100, 200, 250, 300, 400, 500]
for max_length in max_lengths:
    measure_performance(max_length)


'''
MacBook Air M2 16 GB

Max length: 50
Time taken: 6.47 seconds
RAM used: 2435.43 MB
==================================================
Max length: 100
Time taken: 7.18 seconds
RAM used: 1707.02 MB
==================================================
Max length: 200
Time taken: 12.34 seconds
RAM used: 1674.12 MB
==================================================
Max length: 250
Time taken: 15.36 seconds
RAM used: 1601.26 MB
==================================================
Max length: 300
Time taken: 18.11 seconds
RAM used: 1513.04 MB
==================================================
Max length: 400
Time taken: 18.91 seconds
RAM used: 1521.29 MB
==================================================
Max length: 500
Time taken: 29.33 seconds
RAM used: 1554.81 MB
==================================================
'''
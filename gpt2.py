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
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
Max length: 50
Time taken: 3.39 seconds
RAM used: 590.68 MB
==================================================
Max length: 100
Time taken: 3.49 seconds
RAM used: 310.07 MB
==================================================
Max length: 200
Time taken: 5.44 seconds
RAM used: 260.21 MB
==================================================
Max length: 250
Time taken: 6.82 seconds
RAM used: 279.55 MB
==================================================
Max length: 300
Time taken: 7.15 seconds
RAM used: 284.20 MB
==================================================
Max length: 400
Time taken: 9.42 seconds
RAM used: 327.28 MB
==================================================
Max length: 500
Time taken: 11.16 seconds
RAM used: 252.59 MB
==================================================
'''
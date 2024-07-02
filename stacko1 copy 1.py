import json
import requests
from bs4 import BeautifulSoup
import time
import os

def extract_content(url, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract the title
                title = soup.find('h1', itemprop="name", class_="fs-headline1")
                title_text = title.text.strip() if title else ""
                
                # Extract the question body
                question_body = soup.find('div', class_="s-prose js-post-body", itemprop="text")
                question_text = question_body.text.strip() if question_body else ""
                
                # Extract the answer (assuming the first answer is the accepted one)
                answer_body = soup.find_all('div', class_="s-prose js-post-body", itemprop="text")
                answer_text = answer_body[1].text.strip() if len(answer_body) > 1 else ""

                return {
                    "title": title_text,
                    "question": question_text,
                    "answer": answer_text
                }
            elif response.status_code == 429:
                retries += 1
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Received 429 Too Many Requests. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"An error occurred while processing {url}: {str(e)}")
            return None
    print(f"Exceeded max retries for {url}")
    return None

def create_jsonl_entry(content):
    return json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{content['title']}\n{content['question']}"
            },
            {
                "role": "assistant",
                "content": content['answer']
            }
        ]
    })

def get_processed_links(filename):
    processed_links = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                entry = json.loads(line)
                processed_links.add(entry['messages'][1]['content'].split('\n')[0])
    return processed_links

# Load links from JSON file
with open('all_href_links1.json', 'r') as f:
    data = json.load(f)
    links = data['links']

output_file = 'stackoverflow_qa1.jsonl'
processed_links = get_processed_links(output_file)

# Process each link and write to JSONL file
with open(output_file, 'a') as outfile:
    for i, link in enumerate(links, 1):
        if link in processed_links:
            print(f"Skipping already processed link {i}/{len(links)}: {link}")
            continue

        print(f"Processing link {i}/{len(links)}: {link}")
        content = extract_content(link)
        if content:
            jsonl_entry = create_jsonl_entry(content)
            outfile.write(jsonl_entry + '\n')
            outfile.flush()  # Ensure the line is written immediately
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

print("Processing complete. Results saved in stackoverflow_qa.jsonl")

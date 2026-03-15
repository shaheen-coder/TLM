import csv
import re

input_file = "tiny_lm_dataset.txt"  # your raw dataset file
output_file = "eng_convo.csv"

pattern = re.compile(r"<prompt>(.*?)<ai>(.*?)<end>", re.DOTALL)

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

matches = pattern.findall(text)

for prompt, response in matches:
    prompt = prompt.strip()
    response = response.strip()
    rows.append([prompt, response])

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "response"])  # header
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_file}")

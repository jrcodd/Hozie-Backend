from datasets import load_dataset
from mistralai import Mistral
import pandas as pd
import time

# Replace this with your actual Hozie query function
def query_hozie(prompt: str) -> str:
    """Replace with actual call to Hozie system."""
    client = Mistral(api_key=)
    response = client.messages.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Load a subset of MMLU
dataset = load_dataset("hendrycks_test", "high_school_chemistry", split="test[:50]")

    # Placeholder using OpenAI as example
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Load a subset of MMLU
dataset = load_dataset("hendrycks_test", "high_school_chemistry", split="test[:50]")

correct = 0
total = len(dataset)
results = []

for item in dataset:
    question = item['question']
    choices = [item['A'], item['B'], item['C'], item['D']]
    correct_answer = item['answer']

    prompt = f"""Answer the following multiple choice question by letter only (A/B/C/D):

Question: {question}
Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}"""

    try:
        answer = query_hozie(prompt).strip().upper()[0]
    except Exception as e:
        answer = "Error"
        print("Error during query:", e)

    is_correct = answer == correct_answer
    correct += int(is_correct)

    results.append({
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "hozie_answer": answer,
        "is_correct": is_correct
    })

    time.sleep(1)  # Avoid hitting rate limits

# Accuracy
accuracy = correct / total
print(f"\nHozie's Accuracy on High School Chemistry (50 Qs): {accuracy:.2%}")

# Optional: Save results
pd.DataFrame(results).to_csv("hozie_mmlu_results.csv", index=False)

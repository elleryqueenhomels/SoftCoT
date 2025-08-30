import argparse
import os

from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser(description="Prepare Evaluation Datasets")
parser.add_argument("--dataset", type=str, default="xuyige/ASDiv-Aug", help="Dataset to prepare")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

if "ASDiv-Aug" in args.dataset:
  dataset = load_dataset("xuyige/ASDiv-Aug")["test"]
  with open(os.path.join(args.output_dir, 'aug-dev.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['question'], enrty['answer']
      f.write(f'{{"question": "{question}", "answer": "{answer}"}}\n')

elif "AIME_2024" in args.dataset:
  dataset = load_dataset("Maxwell-Jia/AIME_2024")['train']
  with open(os.path.join(args.output_dir, 'aime-2024-train.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['Problem'].replace('\\', '\\\\'), enrty['Answer']
      f.write(f'{{"question": "{question}", "answer": "####{answer}"}}\n')

elif "AIME2025" in args.dataset:
  dataset = concatenate_datasets([
    load_dataset("opencompass/AIME2025", "AIME2025-I")['test'],
    load_dataset("opencompass/AIME2025", "AIME2025-II")['test'],
  ])
  with open(os.path.join(args.output_dir, 'aime-2025-test.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['question'].replace('\\', '\\\\'), enrty['answer']
      f.write(f'{{"question": "{question}", "answer": "####{answer}"}}\n')

elif "MATH-500" in args.dataset:
  dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
  with open(os.path.join(args.output_dir, 'math-500-test.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['problem'].replace('\\', '\\\\'), enrty['answer']
      f.write(f'{{"question": "{question}", "answer": "{answer}"}}\n')

else:
  raise ValueError(f"Unsupported dataset: {args.dataset}")

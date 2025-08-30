import argparse
import json
import os

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Prepare Evaluation Datasets")
parser.add_argument("--dataset", type=str, default="xuyige/ASDiv-Aug", help="Dataset to prepare")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

if "asdiv-aug" in args.dataset.lower():
  dataset = load_dataset("xuyige/ASDiv-Aug")["test"]
  with open(os.path.join(args.output_dir, 'aug-dev.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['question'], enrty['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"{answer}"})
      f.write(json_string + '\n')

elif "aime_2024" in args.dataset.lower():
  dataset = load_dataset("Maxwell-Jia/AIME_2024")['train']
  with open(os.path.join(args.output_dir, 'aime-2024-train.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['Problem'], enrty['Answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"####{answer}"})
      f.write(json_string + '\n')

elif "aime_2025" in args.dataset.lower():
  dataset = load_dataset("MathArena/aime_2025")['train']
  with open(os.path.join(args.output_dir, 'aime-2025-test.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['problem'], enrty['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"####{answer}"})
      f.write(json_string + '\n')

elif "math-500" in args.dataset:
  dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
  with open(os.path.join(args.output_dir, 'math-500-test.jsonl'), 'a') as f:
    for enrty in dataset:
      question, answer = enrty['problem'], enrty['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"{answer}"})
      f.write(json_string + '\n')

else:
  raise ValueError(f"Unsupported dataset: {args.dataset}")

import argparse
import json
import os

from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser(description="Prepare Evaluation Datasets")
parser.add_argument("--dataset", type=str, default="xuyige/ASDiv-Aug", help="Dataset to prepare")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

if "asdiv-aug" in args.dataset.lower():
  file_path = os.path.join(args.output_dir, 'aug-dev.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("xuyige/ASDiv-Aug")["test"]
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer = entry['question'], entry['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"{answer}"})
      f.write(json_string + '\n')
  
  file_path = os.path.join(args.output_dir, 'aug-train.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("xuyige/ASDiv-Aug")["train"]
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer = entry['question'], entry['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"{answer}"})
      f.write(json_string + '\n')

elif "aime_2024" in args.dataset.lower():
  file_path = os.path.join(args.output_dir, 'aime-2024-train.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("Maxwell-Jia/AIME_2024")['train']
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer = entry['Problem'], entry['Answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"####{answer}"})
      f.write(json_string + '\n')

elif "aime2025" in args.dataset.lower():
  file_path = os.path.join(args.output_dir, 'aime-2025-test.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = concatenate_datasets([
    load_dataset("opencompass/AIME2025", "AIME2025-I")['test'],
    load_dataset("opencompass/AIME2025", "AIME2025-II")['test'],
  ])
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer = entry['question'], entry['answer']
      answer = answer.replace('^\circ', '')
      json_string = json.dumps({"question": f"{question}", "answer": f"####{answer}"})
      f.write(json_string + '\n')

elif "math-500" in args.dataset.lower():
  file_path = os.path.join(args.output_dir, 'math-500-test.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer = entry['problem'], entry['answer']
      json_string = json.dumps({"question": f"{question}", "answer": f"{answer}"})
      f.write(json_string + '\n')

elif "strategyqa" in args.dataset.lower():
  file_path = os.path.join(args.output_dir, 'strategyqa_test.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("ChilleD/StrategyQA")["test"]
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer, facts = entry['question'], entry['answer'], entry['facts']
      answer = answer.lower() == 'true'
      json_string = json.dumps({"question": f"{question}", "answer": answer, "facts": f"{facts}"})
      f.write(json_string + '\n')
  
  file_path = os.path.join(args.output_dir, 'strategyqa_train.jsonl')
  if os.path.exists(file_path):
    print(f'>>> Remove old file: {file_path}')
    os.remove(file_path)
  dataset = load_dataset("ChilleD/StrategyQA")["train"]
  with open(file_path, 'a') as f:
    for entry in dataset:
      question, answer, facts = entry['question'], entry['answer'], entry['facts']
      json_string = json.dumps({"question": f"{question}", "answer": answer, "facts": f"{facts}"})
      f.write(json_string + '\n')

else:
  raise ValueError(f"Unsupported dataset: {args.dataset}")

import argparse
import concurrent
import os
import random
import torch
from datasets import load_dataset
from transformers import CodeLlamaTokenizer


"""
Classes
"""
class HFDatasetIterator:
    def __init__(self, dataset_name):
        """
        Class implementing an iterator for iterating over a Dataset from HuggingFace
        :param dataset_name: HuggingFace-Name for Dataset to load data from
        """
        self.dataset_name = dataset_name

        self.LANGS = ["assembly", "batchfile", "cpp", "c", "c-sharp", "cmake", "css", "dockerfile", "fortran", "go",
                      "haskell", "html", "java", "javascript", "julia", "lua", "makefile", "markdown", "perl", "php",
                      "powershell", "python", "ruby", "rust", "scala", "shell", "sql", "tex", "typescript",
                      "visual-basic"]

        """self.LANGS = ["assembly", "batchfile", "c++", "c", "c-sharp", "cmake", "css", "dockerfile", "fortran", "go",
                 "haskell", "html", "java", "javascript", "julia", "lua", "makefile", "markdown", "perl", "php",
                 "powershell", "python", "ruby", "rust", "scala", "shell", "sql", "tex", "typescript", "visual-basic"]"""

        print(">>> Loading Dataset Handles from HuggingFace. This may take a while, depending on number of "
              "Programming Languages specified!")
        self.datasets = [
            load_dataset(self.dataset_name, data_dir=f"data/{lang}", streaming=True, split="train") for lang in self.LANGS
        ]
        self.iters = [iter(item) for item in self.datasets]

    def __next__(self):
        idx = random.randint(0, len(self.datasets) - 1)
        return next(self.iters[idx])

    def __iter__(self):
        return self


parser = argparse.ArgumentParser()

parser.add_argument(
    "num_files",
    type=int,
    help="Number of Files to generate. If Multiprocessing is used: Has to be multiple of num_processes"
)

parser.add_argument(
    "num_samples_per_file",
    type=int,
    help="Number of tokenized Contexts per File"
)
parser.add_argument(
    "--save_path",
    default="./tokenized_dataset",
    type=str,
    help="Path to save-directory for the generated Dataset"
)

parser.add_argument(
    "--min_tokens",
    default=64,
    type=int,
    help="Number of minimum Tokens one Context has to contain in order to be used in the generated Dataset"
)

parser.add_argument(
    "--num_processes",
    default=1,
    type=int,
    help="Number of concurrent Processes for downloading Contexts"
)

parser.add_argument(
    "--tokenizer_name",
    default="codellama/CodeLlama-7b-Instruct-hf",
    type=str,
    help="HuggingFace-Name for the Tokenizer for tokenizing the Contexts. Currently only works for CodeLlama-Tokenizers"
)

parser.add_argument(
    "--dataset_name",
    default="bigcode/the-stack-dedup",
    type=str,
    help="HuggingFace-Name for Dataset to load data from"
)


"""
Parse Arguments
"""
args = parser.parse_args()

NUM_FILES = args.num_files
NUM_SAMPLES_PER_FILE = args.num_samples_per_file
PATH = args.save_path

MIN_TOKENS = args.min_tokens
NUM_PROCESSES = args.num_processes

tokenizer_name = args.tokenizer_name
dataset_name = args.dataset_name


"""
Checks for Arguments
"""
if NUM_FILES % NUM_PROCESSES != 0:
    raise Exception("Argument num_files has to be a multiple of num_processes")

if not os.path.isdir(PATH):
    os.mkdir(PATH)

# Technically also checks, since Errors are raised when strings/names are invalid
tokenizer = CodeLlamaTokenizer.from_pretrained(tokenizer_name)
d = HFDatasetIterator(dataset_name)


def task(id, num_files_per_process):
    # Save all tokenized Contexts of one File in this list
    tokenized = []

    # Generate num_files_per_process Files
    for i_file in range(num_files_per_process):
        print(f"Process: {id} /// File: {i_file} of {num_files_per_process}")

        # Load NUM_SAMPLES_PER_FILE Contexts from HuggingFace Dataset
        while len(tokenized) < NUM_SAMPLES_PER_FILE:
            # Load a single Context from HuggingFace Dataset
            item = next(d)
            tokens = tokenizer(item["content"], return_tensors="pt", truncation=True,
                               max_length=MIN_TOKENS, add_special_tokens=False)

            # If loaded Context from HuggingFace Dataset is too short, reload a new one
            while len(tokens["input_ids"][0]) < MIN_TOKENS:
                item = next(d)
                tokens = tokenizer(item["content"], return_tensors="pt", truncation=True,
                                   max_length=MIN_TOKENS, add_special_tokens=False)

            # Append tokenized Context to list
            tokenized.append(tokens["input_ids"])

            # Give Progress update each time 10% of progress are completed
            if (len(tokenized) % (NUM_SAMPLES_PER_FILE / 10)) == 0:
                print(f"Process: {id} /// {len(tokenized)} of {NUM_SAMPLES_PER_FILE}")

        # Concatenate List, save List and empty list for next File
        tensor = torch.cat(tokenized)
        torch.save(tensor, os.path.join(PATH, f"fragment_{id}_{i_file}.pt"))
        tokenized = []


"""
Start Processes with task
"""
files_per_process = NUM_FILES // NUM_PROCESSES
with concurrent.futures.ProcessPoolExecutor() as executor:
    for i in range(NUM_PROCESSES):
        executor.submit(task, i, files_per_process)

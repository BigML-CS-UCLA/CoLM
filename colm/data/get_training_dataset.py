import random
import copy
import contextlib
from functools import partial
from typing import List, Union, Dict, Sequence
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import transformers
import datasets
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import Dataset

import colm.data.utils as utils


IGNORE_INDEX = -100


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, subset_index_files=None, template_variation=False, seed=0):
    """ get training dataset with a specified seed """
    raw_datasets = load_raw_dataset(
            train_files, sample_percentage=sample_percentage, subset_index_files=subset_index_files, seed=seed)
    
    if "instruction" in raw_datasets.column_names:
        lm_datasets = SupervisedDataset(
            list_data_dict=raw_datasets,
            tokenizer=tokenizer, 
            template_variation=template_variation)
    else:
        lm_datasets = encode_data(
            raw_datasets, tokenizer, max_seq_length)
    
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, subset_index_files=None, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    if len(train_files) == 1 and not train_files[0].endswith(".jsonl"):
        processed_datasets = load_dataset(train_files[0], cache_dir="/data/huggingface_datasets")["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0])
            processed_datasets = processed_datasets.select(subset_indices)
    else:
        processed_datasets = load_dataset(
            "json",
            data_files=train_files,
        )["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0])
            processed_datasets = processed_datasets.select(subset_indices)

    print(f'Before selection, keys are {processed_datasets[0].keys()}')

    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    subset_selection = "use_small_sources"
    if subset_selection == "random":
        with temp_seed(seed):
            index = np.random.permutation(len(processed_datasets))[:sample_size]

        sampled_dataset = processed_datasets.select(index)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "balanced_longest_selection":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example['source']].append((idx, len(example['output'])))

        # Sort examples within each source by output length (descending)
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x[1], reverse=True)

        # Calculate how many samples to take from each source
        num_sources = len(source_groups)
        samples_per_source = sample_size // num_sources
        extra_samples = sample_size % num_sources

        # Select longest samples from each source
        selected_indices = []
        for i, (source, examples) in enumerate(source_groups.items()):
            num_to_select = samples_per_source + (1 if i < extra_samples else 0)
            selected_indices.extend([idx for idx, _ in examples[:num_to_select]])

        # If we don't have enough samples, take whatever is available
        if len(selected_indices) < sample_size:
            remaining = sample_size - len(selected_indices)
            all_remaining = [idx for source in source_groups.values() for idx, _ in source[samples_per_source:]]
            selected_indices.extend(all_remaining[:remaining])

        # Shuffle the selected indices
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "longest_sourcewise_selection":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example['source']].append((idx, len(example['output'])))

        # Sort examples within each source by output length (descending)
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x[1], reverse=True)
        
        # Select longest samples from each source
        selected_indices = []
        for i, (source, examples) in enumerate(source_groups.items()):
            # examples is a list of (idx, length) tuples
            selected_indices.extend([idx for idx, _ in examples[:int(len(examples) * sample_percentage)]])

        # Shuffle the selected indices
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        print(f'Sampled dataset is len {len(sampled_dataset)}, sample size was {sample_size}')
        assert abs(len(sampled_dataset) - sample_size) <= 10
    elif subset_selection == "longest_selection":
        example_indices_and_lengths = [(idx, len(example['output'])) for idx, example in enumerate(processed_datasets)]
        example_indices_and_lengths.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in example_indices_and_lengths[:sample_size]]

        # Shuffle the selected indices
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "use_small_sources":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example['source']].append((idx, len(example['output'])))

        # Sort sources by size
        sorted_sources = sorted(source_groups.items(), key=lambda x: len(x[1]))
        small_sources = sorted_sources[:10]  # Select smallest 10 sources to use fully
        large_sources = sorted_sources[10:]  # Use some X% of the remaining sources

        # Calculate number of samples to take from large sources
        small_sources_total = sum(len(group) for _, group in small_sources)
        large_sources_sample_size = sample_size - small_sources_total
        large_sources_total = sum(len(group) for _, group in large_sources)
        large_source_percentage = large_sources_sample_size / large_sources_total

        selected_indices = []
        
        # Take all samples from small sources
        for _, group in small_sources:
            selected_indices.extend([idx for idx, _ in group])

        # Sort data in each of the large sources by output length and take the longest large_source_percentage% of samples
        for _, group in large_sources:
            group.sort(key=lambda x: x[1], reverse=True)
            group_sample_size = int(len(group) * large_source_percentage)
            group_selected_indices = [idx for idx, _ in group[:group_sample_size]]
            
            selected_indices.extend(group_selected_indices)

        # Shuffle selected indices
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        print(f'Using small sources fully and {int(large_source_percentage*100)}% of large sources')
        assert abs(len(sampled_dataset) - sample_size) <= 5


    return sampled_dataset



def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    print(f'USING ENCODE FUNCTION {encode_function}')
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    return encode_function


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    Used for LESS datasets.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
    
    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
    
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict: datasets.arrow_dataset.Dataset, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        if template_variation:
            PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
        # Change prompt for less datasets
        else:
            PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        targets = []
        data_sources = []
        indices = []
        weights = []
        completion_lengths = [] # Phi-2 tokenizer
        num_empty_output = 0
        all_data_sources = set()
        
        for example in list_data_dict:
            # There is an example with missing output
            # https://github.com/TIGER-AI-Lab/MAmmoTH/issues/36
            if len(example['output']) == 0:
                num_empty_output += 1
                continue
            elif example.get("input", "") != "":
                sources.append(prompt_input.format_map(example))
            else:
                sources.append(prompt_no_input.format_map(example))

            targets.append(f"{example['output']}{tokenizer.eos_token}")
            data_sources.append(example['source'])
            all_data_sources.add(example['source'])
            if 'original_index' in example:
                indices.append(example['original_index'])
            else:
                indices.append(-1)
                
            if 'weight' in example:
                weights.append(example['weight'])
            else:
                weights.append(-1)
            
            if 'completion_length' in example:
                completion_lengths.append(example['completion_length'])
            else:
                completion_lengths.append(-1)
        
        print(f"Discard {num_empty_output} examples")
        
        # Convert data source name to int
        all_data_sources = sorted(list(all_data_sources))
        data_source_to_num = {data_source: idx for idx, data_source in enumerate(all_data_sources)}
        print(data_source_to_num)
        data_sources = [data_source_to_num[data_source] for data_source in data_sources]

        self.sources = sources
        self.targets = targets
        self.all_data_sources = all_data_sources
        self.data_sources = data_sources
        self.indices = indices
        self.weights = weights
        self.completion_lengths = completion_lengths
        self.num_sources = len(data_source_to_num)

    def __len__(self):
        return len(self.sources)
    
    def get_super_class(self, list_small, separate_large=False):
        """
        0: small sources
        1-: large sources
        """
        assert len(list_small) > 0, "Small source needs to contain at least one data source"
        list_super_class = []
        
        # Create a mapping for large sources
        large_source_idx = 1
        large_source_idx_mapping = {}
        
        if separate_large:
            for source in range(self.num_sources):
                if source not in list_small:
                    large_source_idx_mapping[source] = large_source_idx
                    large_source_idx += 1
            
            print(f"Index mapping for large sources: {large_source_idx_mapping}")
        
        for source in self.data_sources:
            if source in list_small:
                list_super_class.append(0)
            elif separate_large:
                # If separate, consider each large source separately
                list_super_class.append(large_source_idx_mapping[source])
            else:
                # If not separate, consider all large sources as one
                list_super_class.append(1)
                
        return list_super_class

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sources[i], 
            labels=self.targets[i], 
            sources=self.data_sources[i],
            indices=self.indices[i],
            weights=self.weights[i],
            completion_lengths=self.completion_lengths[i])

    def __getitem__(self, i):
        return dict(
            input_ids=self.sources[i], 
            labels=self.targets[i], 
            sources=self.data_sources[i],
            indices=self.indices[i],
            weights=self.weights[i],
            completion_lengths=self.completion_lengths[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDatasetWithSource(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_sources = []
        indices = []
        weights = []
        completion_lengths = []
        
        for instance in instances:
            data_sources.append(instance["sources"])
            indices.append(instance['indices'])
            weights.append(instance['weights'])
            completion_lengths.append(instance['completion_lengths'])
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            sources=data_sources,
            indices=indices,
            weights=weights,
            completion_lengths=completion_lengths
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        data_sources = []
        indices = []
        weights = []
        completion_lengths = []
        
        for instance in instances:
            sources.append(instance['input_ids'])
            targets.append(instance['labels'])
            data_sources.append(instance['sources'])
            indices.append(instance['indices'])
            weights.append(instance['weights'])
            completion_lengths.append(instance['completion_lengths'])

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            sources=data_sources,
            indices=indices,
            weights=weights,
            completion_lengths=completion_lengths
        )


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
        
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
    
    
class HFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def convert_superglue_to_hf_source(samples, task, tokenizer, max_length, max_new_tokens, non_diff, train_as_classification, only_train_option):
    """
    Convert samples to HF-compatible dataset
    """
    data = []
    
    for sample in tqdm(samples, mininterval=10):
        encoded_candidates, option_lens = utils.encode_prompt(
            task, 
            task.get_template(), 
            [], 
            sample, 
            tokenizer, 
            max_length=max_length, 
            generation=task.generation, 
            generation_with_gold=True, 
            max_new_tokens=max_new_tokens
        )
        if task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        
        if non_diff:
            # For non-differentiable objective, there is no teacher forcing thus the 
            # current answer part is removed
            encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

        if train_as_classification:
            # For classification, we provide the label as the correct candidate id
            data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates), "sources": sample.data['source']} for _i in range(len(encoded_candidates))])
        elif only_train_option:
            # Otherwise, it is just LM-style teacher forcing
            if non_diff:
                # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate,"sources": sample.data['source']})
            else:
                data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id],"sources": sample.data['source']})
        else:
            data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id],"sources": sample.data['source']})
    
    return data


def convert_superglue_to_hf(samples, task, tokenizer, max_length, max_new_tokens, non_diff, train_as_classification, only_train_option):
    """
    Convert samples to HF-compatible dataset
    """
    data = []
    
    for sample in tqdm(samples, mininterval=10):
        encoded_candidates, option_lens = utils.encode_prompt(
            task, 
            task.get_template(), 
            [], 
            sample, 
            tokenizer, 
            max_length=max_length, 
            generation=task.generation, 
            generation_with_gold=True, 
            max_new_tokens=max_new_tokens
        )
        if task.generation:
            correct_candidate_id = 0
        elif isinstance(sample.correct_candidate, list):
            correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct_candidate_id = sample.candidates.index(sample.correct_candidate)
        
        if non_diff:
            # For non-differentiable objective, there is no teacher forcing thus the 
            # current answer part is removed
            encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

        if train_as_classification:
            # For classification, we provide the label as the correct candidate id
            data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
        elif only_train_option:
            # Otherwise, it is just LM-style teacher forcing
            if non_diff:
                # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
            else:
                data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
        else:
            data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
    
    return data
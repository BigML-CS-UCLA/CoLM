import os
import json
import string
import argparse
from tqdm import tqdm
from collections import Counter
from typing import Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from colm.data.tasks import get_task
from colm.data.utils import encode_prompt


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
# Model
parser.add_argument("--model", default='', type=str)
parser.add_argument("--output_dir", default='', type=str)
parser.add_argument("--dtype", default='bfloat16', type=str)
parser.add_argument("--use_vllm", action='store_true', default=False)
parser.add_argument("--load_8bit", action='store_true', default=False)
parser.add_argument("--cache_dir", default='/data/huggingface_models', type=str)
parser.add_argument("--max_length", default=2048, type=int)
# Data
parser.add_argument("--task", required=True, choices=[
    'SST2', 'Copa', 'BoolQ', 'MultiRC', 'CB', 'WIC', 'WSC', 'ReCoRD', 'RTE', 'SQuAD', 'DROP'], type=str)
parser.add_argument("--seed", default=0, type=int)
# Calibration
parser.add_argument("--sfc", action='store_true', default=False)
parser.add_argument("--icl_sfc", action='store_true', default=False)
# Generation
parser.add_argument("--sampling", action='store_true', default=False)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--num_beams", default=1, type=int)
parser.add_argument("--top_k", default=1, type=int)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--max_new_tokens", default=50, type=int)
parser.add_argument("--eos_token", default="\n", type=str)

args = parser.parse_args()

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]

        
def forward(args, model, tokenizer, input_ids, option_len=None, generation=False):
    """
    Given input_ids and the length of the option, return the log-likelihood of each token in the option.
    For generation tasks, return the generated text.
    This function is only for inference
    """
    input_ids = torch.tensor([input_ids]).cuda()

    if generation:
        # Autoregressive generation
        outputs = model.generate(
            input_ids=input_ids, 
            do_sample=args.sampling, 
            temperature=args.temperature, 
            num_beams=args.num_beams, 
            top_p=args.top_p, 
            top_k=args.top_k, 
            max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
            num_return_sequences=1, 
            eos_token_id=[
                tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], 
                tokenizer.eos_token_id
            ],
        )
        # For generation, directly return the text output
        output_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
        
        return output_text
    else:
        with torch.inference_mode():
            model.eval()
            logits = model(input_ids=input_ids).logits
        labels = input_ids[0, 1:]
        logits = logits[0, :-1] 
        log_probs = F.log_softmax(logits, dim=-1)

        selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
        selected_log_probs = selected_log_probs.cpu().detach()
        # Only return the option (candidate) part
        
        return selected_log_probs[-option_len:]
        

def one_step_pred(args, task, model, tokenizer, train_samples, eval_sample, verbose=False):
    """
    Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
    """
    if verbose:
        print("========= Example =========")
        print(f"Candidate: {eval_sample.candidates}")
        print(f"Correct candidate: {eval_sample.correct_candidate}")


    # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
    encoded_candidates, option_lens = encode_prompt(
        task, 
        task.get_template(), 
        train_samples, 
        eval_sample, 
        tokenizer, 
        max_length=args.max_length, 
        generation=task.generation, 
        max_new_tokens=args.max_new_tokens
    )

    # Calibration
    if args.sfc or args.icl_sfc:
        sfc_encoded_candidates, sfc_option_lens = encode_prompt(
            task, 
            task.get_template(), 
            train_samples, 
            eval_sample, 
            tokenizer, 
            max_length=args.max_length,
            sfc=args.sfc, 
            icl_sfc=args.icl_sfc, 
            generation=task.generation, 
            max_new_tokens=args.max_new_tokens
        )

    outputs = []
    if task.generation:
        # For generation tasks, return the autoregressively-generated text
        output_text = forward(
            args, 
            model, 
            tokenizer, 
            encoded_candidates[0], 
            generation=True)
        if verbose:
            print("=== Prompt ===")
            print(tokenizer.decode(encoded_candidates[0]))
            print(f"Output: {output_text}") 
        return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
    else:
        # For classification/multiple-choice, calculate the probabilities of all candidates
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = forward(
                args, 
                model, 
                tokenizer, 
                encoded_candidate, 
                option_len=option_lens[candidate_id])
            if verbose:
                if candidate_id == 0:
                    print("=== Candidate %d ===" % candidate_id)
                    print(tokenizer.decode(encoded_candidate))
                else:
                    print("=== Candidate %d (without context)===" % candidate_id)
                    print(tokenizer.decode(encoded_candidate).split(task.train_sep)[-1])
                print(f"Log probabilities of the option tokens: {selected_log_probs}")

            if args.sfc or args.icl_sfc:
                sfc_selected_log_probs = forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                if verbose:
                    print("=== Candidate %d (without context) SFC ===" % candidate_id)
                    print(tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(task.train_sep)[-1])
                    print(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

            outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if args.sfc or args.icl_sfc else None})

        if args.sfc or args.icl_sfc:
            # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
            # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
            scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
        else:
            # (Default) length-normalized log probabilities
            # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
            scores = [x['log_probs'].mean().item() for x in outputs]

        if verbose:
            print(f"Prediction scores: {scores}")

        if isinstance(eval_sample.correct_candidate, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))
        
        
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))       

        
def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
        else:
            return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])
    elif metric_name == "em":
        # For question answering
        return np.mean([any([normalize_answer(ans) == normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate]) for pred in predictions])
    elif metric_name == "f1":
        # For question answering
        f1 = []
        for pred in predictions:
            all_f1s = []
            if pred.correct_candidate[0] == "CANNOTANSWER" or pred.correct_candidate[0] == "no answer":
                f1.append(int(normalize_answer(pred.correct_candidate[0]) == normalize_answer(pred.predicted_candidate)))
            else:
                for ans in pred.correct_candidate:
                    prediction_tokens = normalize_answer(pred.predicted_candidate).split()
                    ground_truth_tokens = normalize_answer(ans).split()
                    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                    num_same = sum(common.values())
                    if num_same == 0:
                        all_f1s.append(0)
                    else:
                        precision = 1.0 * num_same / len(prediction_tokens)
                        recall = 1.0 * num_same / len(ground_truth_tokens)
                        all_f1s.append((2 * precision * recall) / (precision + recall))
                f1.append(max(all_f1s))

        return np.mean(f1)
        
        
def evaluate(args, task, model, tokenizer, train_samples, eval_samples, one_train_set_per_eval_sample=False):
    """
    Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
    """
    if one_train_set_per_eval_sample:
        print(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
    else:
        print(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

    # Prediction loop
    predictions = []  
    for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
        predictions.append(
            one_step_pred(
                args,
                task,
                model,
                tokenizer,
                train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, 
                eval_sample, 
                verbose=False)
        )

    # Calculate metrics 
    metric_name = getattr(task, "metric_name", "accuracy")
    metrics = {metric_name: calculate_metric(predictions, metric_name)}
    
    return metrics


def main():
    # Set up the model
    is_peft = os.path.exists(os.path.join(
        args.model, "adapter_config.json"))
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        trust_remote_code=True,
        cache_dir=args.cache_dir)
    
    if args.use_vllm:
        raise ValueError("SuperGLUE evaluation with vLLM is not supported yet.")
    else:
        if is_peft:
            # load this way to make sure that optimizer states match the model structure
            config = LoraConfig.from_pretrained(args.model)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, 
                torch_dtype="auto", 
                device_map="auto", 
                cache_dir=args.cache_dir)
            model = PeftModel.from_pretrained(
                base_model, 
                args.model, 
                device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                load_in_8bit=args.load_8bit,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=args.cache_dir)
        model.eval()
        
        # pad token is not added by default for pretrained models
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            
    # Get task and data
    task = get_task(args.task)
    # train_samples = task.sample_subset(data_split="train", seed=args.seed, num=-1)
    train_samples = []
    if args.task in ["ReCoRD", "Copa"]: 
        # Set eval limit for multiple choice to reduce inference time
        eval_samples = task.sample_subset(data_split="valid", seed=args.seed, num=1000)
    elif args.task in ["SQuAD", "DROP"]:
        # Set eval limit for generation to reduce inference time
        eval_samples = task.sample_subset(data_split="valid", seed=args.seed, num=1000)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    else:
        eval_samples = task.sample_subset(data_split="valid", seed=args.seed, num=-1)

    results = evaluate(
        args,
        task,
        model,
        tokenizer,
        train_samples, 
        eval_samples, 
        one_train_set_per_eval_sample=False)

    # Print and save results
    print(f"Results for {args.task}:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
    # Save results to file
    if not args.output_dir:
        output_dir = f"{args.model}/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{args.task}_eval_results.json"
    else:
        output_file = f"{args.output_dir}/{args.task}_eval_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
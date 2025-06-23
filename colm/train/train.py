#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import json

import datasets
import torch
import torch.distributed as dist
import transformers
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoConfig,
    PhiConfig,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from colm.data.get_training_dataset import (
    convert_superglue_to_hf,
    convert_superglue_to_hf_source,
    get_training_dataset,
    SupervisedDataset,
    HFDataset,
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithSource)
from colm.data.tasks import get_task, Sample
from colm.data.utils import (
    forward_wrap_with_option_len,
    NondiffCollator,
    DataCollatorWithPaddingAndNesting
)
from colm.train.huggingface_trainer import CustomTrainer as Trainer
from colm.train.subset_trainer_distributed import SubsetTrainer, SubsetTrainerEfficient
from colm.train.data_arguments import DataArguments, get_data_statistics
from colm.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from colm.train.training_arguments import TrainingArguments
from colm.train.custom_phi import DecomposedPhiCausalLM


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DTYPES = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'auto': 'auto',
    'none': None
}


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # Set run_name for wandb
        training_args.run_name = training_args.output_dir.split('/')[-1]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length)

    if not model_args.enable_dropout:
        # Set dropout to 0
        logger.info("Set dropout to 0")
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        assert isinstance(
            model_config, PhiConfig), "Only support no dropout for Phi-2!"
        model_config.resid_pdrop = 0
        model_args.lora_dropout = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            torch_dtype=DTYPES[model_args.torch_dtype],
            trust_remote_code=True,
            cache_dir=model_args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=DTYPES[model_args.torch_dtype],
            trust_remote_code=True,
            cache_dir=model_args.cache_dir)

    if len(training_args.fsdp) > 0 and training_args.fsdp_config.get('activation_checkpointing', False):
        # Enable gradient checkpointing for reducing memory footprint
        # Bug in future torch version
        # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/discussions/12
        logger.info("Enable gradient checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': True})
    add_padding_to_tokenizer(tokenizer)

    # Resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    modules_to_save = []
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False
        # Adding additional tokens to vocabulary
        # https://github.com/huggingface/peft/issues/334
        modules_to_save = ["lm_head", "embed_tokens"]

    # Set up LoRA
    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, lora_config)
        # ValueError: Attempting to unscale FP16 gradients
        # https://github.com/huggingface/peft/issues/341
        model.base_model.model.model.embed_tokens.weight.data = model.base_model.model.model.embed_tokens.weight.data.float()
        model.base_model.model.lm_head.weight.data = model.base_model.model.lm_head.weight.data.float()
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Change last layer to LoRA
        training_args.last_layers = [
            name + '.lora_B' for name in training_args.last_layers]

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)
    # Load training dataset
    if 'superglue' in data_args.train_files[0]:
        task_name = data_args.train_files[0].split('-')[-1]
        task = get_task(task_name)
        if data_args.train_files[0].split('-')[0] == "load":
            with open('/data/' + f'{data_args.train_files[0]}.jsonl', 'r') as f:
                train_samples = [Sample(**json.loads(line)) for line in f.readlines()]
            print("Load Successfully")
        else:
            train_samples = task.sample_subset(num=1000)
        if (training_args.source_wise_selection != "none"):
            train_dataset = HFDataset(convert_superglue_to_hf_source(
                train_samples,
                task,
                tokenizer=tokenizer,
                max_length=model_args.model_max_length,
                max_new_tokens=training_args.max_new_tokens,
                non_diff=training_args.non_diff,
                train_as_classification=task.classification,
                only_train_option=training_args.only_train_option
            ))
        else:
            train_dataset = HFDataset(convert_superglue_to_hf(
                train_samples,
                task,
                tokenizer=tokenizer,
                max_length=model_args.model_max_length,
                max_new_tokens=training_args.max_new_tokens,
                non_diff=training_args.non_diff,
                train_as_classification=task.classification,
                only_train_option=training_args.only_train_option
            ))
        logger.info(
            f'Train dataset of task {task_name} has {len(train_samples)} examples with attributes generation = {task.generation} and classification = {task.classification}')
        logger.info(f'TRAIN DATASET EXAMPLE: {train_samples[0]}')

        analysis_dataset = None
        if training_args.analysis_mode:
            analysis_dataset = HFDataset(convert_superglue_to_hf(
                task.samples["valid"],
                task,
                tokenizer=tokenizer,
                max_length=model_args.model_max_length,
                max_new_tokens=training_args.max_new_tokens,
                non_diff=training_args.non_diff,
                train_as_classification=task.classification,
                only_train_option=training_args.only_train_option
            ))

        # Change forward pass of model for SuperGLUE
        if training_args.only_train_option and not training_args.non_diff:
            training_args.modify_forward = True
            model.original_forward = model.forward
            model.forward = forward_wrap_with_option_len.__get__(
                model, type(model))

        # Get data collator
        if task.classification:
            data_collator = DataCollatorWithPaddingAndNesting(
                tokenizer, pad_to_multiple_of=8)
        elif training_args.non_diff:
            data_collator = NondiffCollator(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer, pad_to_multiple_of=8)
    else:
        # Change forward pass of model for efficient zeroth-order gradient
        if training_args.data_selection_unit == "mezo" and training_args.efficient_mezo:
            model.decomposer = DecomposedPhiCausalLM(model.model)
        train_dataset = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            subset_index_files=data_args.subset_index_files,
            seed=data_args.sample_data_seed)

        logger.info(f'TRAIN DATASET: {train_dataset[0].keys()}')
        logger.info(f'TRAIN DATASET EXAMPLE: {train_dataset[0]}')

        # Get data collator
        if isinstance(train_dataset, SupervisedDataset):
            if (training_args.source_wise_selection != "none") or (not training_args.remove_unused_columns):
                data_collator = DataCollatorForSupervisedDatasetWithSource(
                    tokenizer=tokenizer)
            else:
                data_collator = DataCollatorForSupervisedDataset(
                    tokenizer=tokenizer)
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding="longest")

        get_data_statistics(train_dataset, is_custom_dataset=isinstance(
            train_dataset, SupervisedDataset))

        if "features" in train_dataset and "dataset" in train_dataset.features:
            train_dataset = train_dataset.remove_columns(
                ["dataset", "id", "messages"])

        analysis_dataset = None
        if training_args.analysis_mode:
            from colm.data.get_validation_dataset import get_dataset
            analysis_dataset = get_dataset(
                training_args.analysis_dataset,
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                max_length=data_args.max_seq_length)

    logger.info(f"Using data collator {type(data_collator)}")

    if len(training_args.keep_sources) and isinstance(data_collator, DataCollatorForSupervisedDatasetWithSource):
        training_args.keep_sources = [
            int(source_idx) for source_idx in training_args.keep_sources.split('_')]
        logger.info(
            "Keep all examples of the following sources in the mini-batch.")

        for source_idx in training_args.keep_sources:
            logger.info(train_dataset.all_data_sources[source_idx])
    else:
        training_args.keep_sources = []

    logger.info(f"Keep source indices in {training_args.keep_sources}")

    # If the actual train batch size is smaller than the data loader batch size
    kwargs = {}
    kwargs["logger"] = logger

    if training_args.data_selection_method == "none":
        logger.info("Using HuggingFace Trainer")
        trainer_class = Trainer
    elif training_args.efficient_mezo:
        logger.info("Using SubsetTrainerEfficient")
        trainer_class = SubsetTrainerEfficient
    else:
        logger.info("Using SubsetTrainer")
        trainer_class = SubsetTrainer

    # Setup wandb
    os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    os.environ["WANDB_NAME"] = training_args.run_name + f'_{os.uname()[1]}'
    os.environ["WANDB_NOTES"] = training_args.wandb_notes
    logger.info('Finished wandb setup.')

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **kwargs
    )

    # Training
    train_result = trainer.train(
        resume_from_checkpoint=model_args.checkpoint_path)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {max_mem_gb:.2f} GB")

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()

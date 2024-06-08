import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)
import wandb

from .arguments import ModelArguments, DataArguments
from .data import TrainDatasetForCE, EvalDatasetForCE, GroupCollator
from .modeling import CrossEncoder
from .trainer import CETrainer

logger = logging.getLogger(__name__)

def main():
    # Initialize wandb
    wandb.init(project="FlagEmbedding", entity="YOUR_WANDB_ENTITY")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levellevel - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="classification",
        cache_dir=model_args.cache_dir,
    )

    model = CrossEncoder.from_pretrained(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    eval_dataset = EvalDatasetForCE(data_args, tokenizer=tokenizer) if data_args.eval_data else None

    trainer = CETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluate ***")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Eval results: {eval_result}")

if __name__ == "__main__":
    main()

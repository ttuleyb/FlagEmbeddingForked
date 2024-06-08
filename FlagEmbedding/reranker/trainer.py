import logging
import os
from typing import Optional, Dict

import torch
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
import wandb

from .modeling import CrossEncoder

logger = logging.getLogger(__name__)

class CETrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        self.log(output.metrics)
        self._log_eval_results(output.metrics)  # Ensure this is called to log evaluation metrics
        return output.metrics

    def prediction_loop(self, dataloader, description, prediction_loss_only=None):
        model = self.model
        model.eval()
        eval_losses = []
        for inputs in dataloader:
            loss = self.compute_loss(model, inputs)
            eval_losses.append(loss.item())
        eval_loss = sum(eval_losses) / len(eval_losses)
        return PredictionOutput(predictions=None, label_ids=None, metrics={"eval_loss": eval_loss})

    def _log_eval_results(self, metrics):
        """
        Log evaluation results to both the console and wandb.
        """
        logger.info("***** Evaluation results *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")
        if wandb.run is not None:
            wandb.log(metrics)

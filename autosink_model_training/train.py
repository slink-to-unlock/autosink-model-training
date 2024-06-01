
import os
import numpy as np
from transformers import Trainer, TrainingArguments, default_data_collator, TrainerCallback
import evaluate


def get_compute_metrics_fn(metric: str):
    __metric = evaluate.load(metric)

    def compute_metrics_fn(p):
        return __metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )

    return compute_metrics_fn


class AIsinkTrainer:
    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.setup_environment()
        self.training_args = self.get_training_arguments()
        self.trainer = self.get_trainer()

    def setup_environment(self):
        os.environ["WANDB_PROJECT"] = "AIsink-resent50"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"



    def get_training_arguments(self):
        return TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            report_to="wandb",  # W&B에 로깅
            run_name='ai-sink-run',
            save_strategy="epoch",  # 매 epoch마다 모델 저장
            save_total_limit=5,  # 최대 5개의 체크포인트 저장
            load_best_model_at_end=True,
        )

    def get_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.get_compute_metrics_fn('accuracy'),
            # callbacks=[MyWandbCallback()]  # W&B 콜백 추가
        )

    def train(self):
        self.trainer.train()

if __name__ == '__main__':
    pass
    # ai_sink_trainer = AIsinkTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    # ai_sink_trainer.train()

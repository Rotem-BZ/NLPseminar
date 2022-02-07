import numpy as np
import torch
from numpy import random

from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from transformers import BertTokenizer, BertModel, BertForTokenClassification

from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.optim import Adam, SGD

from data_utils import MLM_Dataset, FSM_Dataset, CoLaDataset, AmazonDataset

from collections import defaultdict

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

import pytorch_lightning as pl
from operator import itemgetter
from argparse import Namespace

from typing import List, Optional



class GeneralDataModule(pl.LightningDataModule):
    """
    This module is used by Lightining to train the model. It generates the data using the functions x_dataloader.
    """
    def __init__(self, text_tokenizer, dataset_type, train_args, test_args, hparams: Namespace):
        super().__init__()
        self.train_batch_size = hparams.tune_batch_size if test_args else hparams.train_batch_size
        self.tokenizer = text_tokenizer
        self.dataset_type = dataset_type
        self.train_args = train_args
        self.test_args = test_args
        self.hparams.update(vars(hparams))
        self.num_samples = 0
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        print(f"STAGE = {stage}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = self.dataset_type(tokenizer=self.tokenizer, **self.train_args)

            if self.test_args:
                self.val_set = self.dataset_type(tokenizer=self.tokenizer, **self.test_args)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = self.dataset_type(tokenizer=self.tokenizer, **self.test_args)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if not self.test_args:
            return None
        return DataLoader(self.val_set, batch_size=self.hparams.test_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=4)


class SeminarBert(pl.LightningModule):
    """
    This class is used by Lightining to train the model. It holds the base model, which is the BERT architecture given
    in __init__. Implements calculation for variables we want to track - loss, accuracy & std of predictions.
    """
    def __init__(self, base_model, hparams, run_name='', train_losses=None, test_losses=None):
        super(SeminarBert, self).__init__()
        # self.save_hyperparameters()
        # self.automatic_optimization = False
        # self.hparams = AttributeDict()
        self.base_model = base_model
        # self.forwards = 0
        # self.dataset = dataset
        self.run_name = run_name
        self.n_correct = defaultdict(int)
        self.all_samples = defaultdict(int)
        self.learning_rate = hparams.lr
        self.hparams.update(vars(hparams))

        # print(f"init----device: {self.device} gpu: {show_gpu(str(self.device))}")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch_data, batch_idx):
        # print('start')
        input_ids = batch_data['input_ids']
        # print(tokenizer.decode(input_ids[0])[:200])
        attention_mask = batch_data['attention_mask']
        # print(f'BATCH SIZE: {input_ids.shape[0]}')
        labels = batch_data['labels']
        # print(f"labels: {labels.shape} inputs: {input_ids.shape}")
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # print(outputs.logits[:, :20, :])
        # print(f"step loss {loss}")
        # std = outputs.logits[labels != -100].std().item()
        predictions = torch.argmax(outputs.logits, dim=-1)

        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        predictions = predictions.flatten()
        labels = labels.flatten()
        # print(f'pred: {predictions[:20]} shape {predictions.shape} max {predictions.max().item()}')
        # print(f'label {labels[:20]} max {labels.max().item()}')
        # print(f'logits shape {outputs.logits.shape}')
        batch_accuracy = (predictions == labels).sum() / len(labels)

        # print(f"standard deviation: {std}")
        self.logger.experiment[f"{self.run_name}/train/acc"].log(batch_accuracy.item())
        # self.logger.experiment[f"{self.run_name}/train/std"].log(std)
        self.logger.experiment[f"{self.run_name}/train/loss"].log(loss)
        # self.logger.experiment[f"train/{self.run_name}/task{i}_std"].log(torch.std(out[:, 1].detach()).item())
        return {'loss': loss, 'accuracy': batch_accuracy}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        acc_ten = torch.stack([out['accuracy'] for out in outputs])
        self.logger.experiment[f"{self.run_name}/train/epoch_acc"].log(acc_ten.mean().item())

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # print(f'step size {labels.shape[0]}')
        outputs = self(input_ids, attention_mask, labels)
        step_outputs = {'logits': outputs.logits, 'loss': outputs.loss, 'labels': labels}
        return step_outputs
        # self.test_losses[t](task_probs, labels[:, t], batch_domains)

    def test_step_end(self, batch_parts) -> Optional[STEP_OUTPUT]:
        logits = batch_parts['logits']
        labels = batch_parts['labels']
        # print(f'step end size {labels.shape[0]}')
        predictions = torch.argmax(logits, dim=-1)
        batch_accuracy = (predictions == labels).sum() / labels.shape[0]
        # return {'acc': batch_accuracy, 'loss': batch_loss}
        return torch.stack((batch_accuracy, batch_parts['loss']))

    def test_epoch_end(self, outputs: list) -> None:
        results = torch.stack(outputs)
        self.logger.experiment[f"{self.run_name}/test/epoch_loss"].log(results[:, 1].mean().item())
        self.logger.experiment[f"{self.run_name}/test/epoch_acc"].log(results[:, 0].mean().item())

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_step_end(self, batch_parts) -> Optional[STEP_OUTPUT]:
        return self.test_step_end(batch_parts)

    def validation_epoch_end(self, outputs: list) -> None:
        self.test_epoch_end(outputs)

    def configure_optimizers(self):
        print(f'CHOSEN LR: {self.learning_rate}')
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def total_steps(self):
        return (self.hparams.train_samples_to_load // self.hparams.train_samples_per_dom_batch) * self.hparams.epochs


def pretrain_func(method: str, hparams: Namespace, logger, tokenizer, model_path, save_to_path: str = None):
    """
    The main pretraining function.
    :param method: Wheter to use MLM or Shuffle+Random
    :param hparams: parameters for training
    :param logger: logger module for Neptune
    :param tokenizer: pretrained tokenizer
    :param model_path: Starting point for the pretraining.
    :param save_to_path: Where to save the pretrained model
    """
    assert method in ['MLM', 'FSM']
    if method == 'MLM':
        base_model = BertForMaskedLM.from_pretrained(model_path)
        dataset_type = MLM_Dataset
    else:
        base_model = BertForTokenClassification.from_pretrained(
            model_path,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False
        )
        dataset_type = FSM_Dataset
    data_module = GeneralDataModule(text_tokenizer=tokenizer, dataset_type=dataset_type,
                                    train_args=hparams.pretrain_args, test_args=dict(), hparams=hparams)
    model = SeminarBert(base_model, hparams, run_name=f'{method} pretrain')
    trainer = pl.Trainer(gpus=hparams.gpus, auto_lr_find=False, max_epochs=hparams.epochs,
                         max_time=hparams.train_max_time,
                         replace_sampler_ddp=False, logger=logger, precision=16, auto_scale_batch_size=False,
                         strategy='ddp', enable_checkpointing=False)
    trainer.fit(model, data_module)
    if save_to_path:
        model.base_model.save_pretrained(save_to_path)


def finetune_func(dataset: str, model_path: str,
                  hparams: Namespace, logger, tokenizer, save_to_path: str = None, cola_in_domain: bool = None):
    """
    The main finetuning function.
    :param dataset: either CoLA or Amazon
    :param model_path: path for initial model - this is how we choose between the differently pretrained models.
    :param hparams: parameters for finetuning
    :param logger: logger module for Neptune
    :param tokenizer: pretrained tokenizer
    :param save_to_path: if not None, where to save the finetuned model.
    :param cola_in_domain: If dataset = CoLA, which domain to test on.
    """
    assert dataset in ['CoLA', 'Amazon']
    if dataset == 'CoLA':
        hparams.test_args['phase'] = 'test_in' if cola_in_domain else 'test_out'

    if dataset == 'Amazon' and hparams.finetune_args['load_n_samples'] == -1:
        hparams.finetune_args['load_n_samples'] = 1500
    if dataset == 'Amazon' and hparams.test_args['load_n_samples'] == -1:
        hparams.test_args['load_n_samples'] = 5000
    if dataset == 'Amazon':
        hparams.finetune_args.update({'labels_feature': 'overall', 'data_feature': 'reviewText'})
        hparams.test_args.update({'labels_feature': 'overall', 'data_feature': 'reviewText'})

    base_model = BertForSequenceClassification.from_pretrained(model_path,
                                                               output_attentions=False,
                                                               output_hidden_states=False)

    dataset_type = {'CoLA': CoLaDataset, 'Amazon': AmazonDataset}[dataset]
    data_module = GeneralDataModule(text_tokenizer=tokenizer, dataset_type=dataset_type,
                                    train_args=hparams.finetune_args, test_args=hparams.test_args, hparams=hparams)
    phase = hparams.test_args['phase']
    model = SeminarBert(base_model, hparams, run_name=f'{dataset} {phase} finetune')
    trainer = pl.Trainer(gpus=hparams.gpus, auto_lr_find=False, max_epochs=hparams.epochs,
                         max_time=hparams.tune_max_time,
                         replace_sampler_ddp=False, logger=logger, precision=16, auto_scale_batch_size=False,
                         enable_checkpointing=False, check_val_every_n_epoch=1)
    trainer.fit(model, data_module)
    print(f"tune size: {len(data_module.train_set)} test size {len(data_module.val_set)}")
    if save_to_path:
        model.base_model.save_pretrained(save_to_path)
    # trainer.test(model, data_module)
    # if dataset == 'CoLA':
    #     hparams.test_args['phase'] = 'test_out'
    #     data_module = GeneralDataModule(text_tokenizer=tokenizer, dataset_type=dataset_type,
    #                                     train_args=hparams.finetune_args, test_args=hparams.test_args, hparams=hparams)
    #     trainer.test(model, data_module)


def pretrain_and_finetune(pretrain_method: str, finetune_datasets: list, pretrain_path: str, final_path: str = None):
    """
    continue BERT pretraining according to 'pretrain_method', and finetune on 'finetune_dataset'
    :param pretrain_method: the pretraining method, one of [MLM, FSM, bert-base-uncased]
    :param finetune_datasets: list of datasets to finetune on. each is either CoLA or Amazon
    :param pretrain_path: path to save pretraining checkpoint in case pretrain_method is in [MLM, FSM]
    :param final_path: path to save final checkpoint (after fine-tuning). if None, there is no final save.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logger = NeptuneLogger(
        project="nivkook9/SeminarBert",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MjlmZDIyNy0yNzYzLTRmNjYtODkxZC0zOGYyMjAxNWE4NzcifQ==",
        # format "<WORKSPACE/PROJECT>"
        log_model_checkpoints=False
    )
    hparams = Namespace(
        lr=1e-5,
        epochs=100,
        train_batch_size=5,
        test_batch_size=10,
        tune_batch_size=10,
        data_path='',
        pretrain_args={'load_n_samples': -1, 'phase': 'train'},
        finetune_args={'load_n_samples': 200, 'phase': 'train'},
        test_args={'load_n_samples': -1, 'phase': 'test'},
        train_max_time={'minutes': 30},
        tune_max_time={'minutes': 3},
        gpus=[2]
    )
    if pretrain_method in ['MLM', 'FSM']:
        pretrain_func(pretrain_method, hparams, logger, tokenizer, 'bert-base-uncased', pretrain_path)
    else:
        pretrain_path = 'bert-base-uncased'
    for i, finetune_dataset in enumerate(finetune_datasets):
        if final_path:
            current_final_path = final_path + '_' + finetune_dataset
        else:
            current_final_path = None
        if finetune_dataset == 'CoLA':
            print(f'PRETRAINED: {pretrain_path}')
            finetune_func(finetune_dataset, pretrain_path, hparams, logger, tokenizer, current_final_path, True)
            finetune_func(finetune_dataset, pretrain_path, hparams, logger, tokenizer, current_final_path, False)
        if finetune_dataset == 'Amazon':
            finetune_func(finetune_dataset, pretrain_path, hparams, logger, tokenizer, current_final_path)

if __name__ == '__main__':
    pretrain_and_finetune(pretrain_method='MLM', finetune_datasets=['CoLA', 'Amazon'],
                          pretrain_path="bert-base-uncased",
                          final_path=None)

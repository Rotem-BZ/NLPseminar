import gzip
import json
import math
import time
import numpy as np
import pandas as pd
import os
import torch
from torch import tensor, Tensor
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets import load_dataset

AMAZON_TRAIN_PATH = '/data/user-data/niv.ko/Amazon_Data/Video_Games.gz'
AMAZON_TEST_PATH = '/data/user-data/niv.ko/Amazon_Data/Office_Products.gz'

class MLM_Dataset(Dataset):
    """
    This class creates data for the MLM objective. Passes each sentence through tokenizer, chooses tokens to mask, and
    replaces these tokens with either the <mask> token or a random token. Labels are the original tokens.
    """
    def __init__(self, phase: str, tokenizer, load_n_samples):
        t0 = time.perf_counter()
        assert phase in ['train', 'test']
        data_path = os.path.join('/data/user-data/niv.ko/BookCorpus', 'bookcorpus_subdataset_2')
        with open(data_path, 'r') as data_file:
            lines = data_file.read()
        lines = lines.split('\n')[:load_n_samples]
        data = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt').data
        input_tensor = data['input_ids']
        now = time.perf_counter()
        print("time to get input tensor:", now - t0)
        self.tokenizer_start_idx = 999
        self.vocab_size = len(tokenizer)
        self.mask_id = tokenizer.mask_token_id
        self.bad_token_ids = tokenizer.all_special_ids
        self.attention_mask = data['attention_mask']
        original_inputs = input_tensor.detach().clone()
        self.labels = torch.ones_like(original_inputs) * -100
        rand = torch.rand(input_tensor.shape)
        # 80% of 15% of words are turned to <mask>
        mask_arr = (rand < 0.15*0.8)
        for special_id in self.bad_token_ids:
            mask_arr *= (input_tensor != special_id)
        # 10% of 15% of words are turned to a random token
        random_arr = (rand < 0.15*0.9)
        for special_id in self.bad_token_ids:
            random_arr *= (input_tensor != special_id)
        input_tensor[random_arr] = torch.randint(low=self.tokenizer_start_idx, high=self.vocab_size,
                                                 size=(random_arr.sum().item(),))
        input_tensor[mask_arr] = self.mask_id
        self.labels[mask_arr] = original_inputs[mask_arr]
        self.input_tensor = input_tensor


    def __getitem__(self, item):
        return {'input_ids': self.input_tensor[item],
                'attention_mask': self.attention_mask[item],
                'labels': self.labels[item]}

    def __len__(self):
        return self.input_tensor.shape[0]


class FSM_Dataset:
    """
    This class creates data for the Shuffle+Random objective. Passes each sentence through tokenizer, shuffles a
    fraction of the tokens, replaces some of the tokens with a random token. Labels are 'original', 'shuffle', 'random'.
    """
    label_to_idx = {'original': 0, 'shuffled': 1, 'random': 2}
    idx_to_label = {0: 'original', 1: 'shuffled', 2: 'random'}

    def __init__(self, phase: str, tokenizer, load_n_samples):
        assert load_n_samples < 20_000, "maximum bookcorpus size (19,999) exceeded"
        t0 = time.perf_counter()
        assert phase in ['train', 'test']
        data_path = os.path.join('/data/user-data/niv.ko/BookCorpus', 'bookcorpus_subdataset_2')
        with open(data_path, 'r') as data_file:
            lines = data_file.read()
        lines = lines.split('\n')[:load_n_samples]
        data = tokenizer(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt').data
        input_ids = data['input_ids']
        now = time.perf_counter()
        print("time to get input tensor:", now - t0)
        labels = torch.ones_like(input_ids) * self.label_to_idx['original']
        labels[data['attention_mask'] == 0] = -100

        self.tokenizer_start_idx = 999
        self.vocab_size = len(tokenizer)
        self.mask_id = tokenizer.mask_token_id
        self.bad_token_ids = tokenizer.all_special_ids

        rand = torch.rand(input_ids.shape)
        # shuffle the words with rand < 0.1, randomize the words with 0.1 < rand < 0.2
        shuffle_choice = (rand < 0.1)
        for special_id in self.bad_token_ids:
            shuffle_choice *= (input_ids != special_id)
        if shuffle_choice.sum() >= 2:
            new_perm_indices = torch.randperm(shuffle_choice.sum().item())
            input_ids[shuffle_choice] = input_ids[shuffle_choice][new_perm_indices]
            labels[shuffle_choice] = self.label_to_idx['shuffled']

        random_choice = (rand < 0.2) * (rand > 0.1)
        for special_id in self.bad_token_ids:
            random_choice *= (input_ids != special_id)
        input_ids[random_choice] = torch.randint(low=self.tokenizer_start_idx, high=self.vocab_size,
                                                 size=(random_choice.sum().item(),))
        labels[random_choice] = self.label_to_idx['random']
        self.input_tensor = input_ids
        self.attention_mask = data['attention_mask']
        self.labels = labels

    def __getitem__(self, item):
        return {'input_ids': self.input_tensor[item],
                'attention_mask': self.attention_mask[item],
                'labels': self.labels[item]}

    def __len__(self):
        return self.input_tensor.shape[0]


class CoLaDataset:
    """
    This class parses the CoLA dataset. The file pathes of the data are stated in the "name" variable in __init__ and
    their folder is '/data/user-data/niv.ko/CoLA/raw'.
    """
    def __init__(self, phase: str, tokenizer, load_n_samples):
        """
        Create samples and labels from the data.
        :param phase: one of ['train, 'test_in', 'test_out']. Indicates the path from which to take the data, with
        test_in being in-domain of the training samples and test_out is out of domain.
        :param tokenizer: A pretrained tokenizer.
        :param load_n_samples: How many samples to load. Value of -1 means load all the samples.
        """
        t0 = time.perf_counter()
        assert phase in ['train', 'test_in', 'test_out']
        name = {'train': 'in_domain_train.tsv', 'test_out': 'out_of_domain_dev.tsv', 'test_in': 'in_domain_dev.tsv'}[phase]
        data_path = os.path.join('/data/user-data/niv.ko/CoLA/raw', name)
        df = pd.read_csv(data_path, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])
        if load_n_samples != -1:
            df = df.head(load_n_samples)
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))
        sentences = df.sentence.values
        labels = df.label.values
        data = tokenizer(sentences.tolist(), max_length=512, padding='max_length', truncation=True, return_tensors='pt').data

        self.input_tensor = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = tensor(labels)

        print("time to init cola dataset:", time.perf_counter() - t0)

    def __getitem__(self, item):
        return {'input_ids': self.input_tensor[item],
                'attention_mask': self.attention_mask[item],
                'labels': self.labels[item]}

    def __len__(self):
        return self.input_tensor.shape[0]

class AmazonDataset(Dataset):
    """
    This class parses the Amazon reviews dataset. The file path of the data is determined by the phase given in
    __init__. This class implements rebalancing of the data.
    """
    def __init__(self, load_n_samples: int, data_feature, labels_feature, tokenizer, phase,
                 balanced=True):
        self.inputs = None
        self.data_feature = data_feature
        self.labels_feature = labels_feature
        self.tokenizer = tokenizer
        self.num_samples = 0
        self.balanced = balanced
        self.domains_tasks = [0]
        data_path = AMAZON_TRAIN_PATH if phase == 'train' else AMAZON_TEST_PATH
        self.load_samples(data_path, load_n_samples)

    def load_samples(self, domain_path, n_samples):
        content = self.read_json(domain_path, n_samples)
        # content = [obj for obj in content if self.data_feature in obj]
        texts = [obj[self.data_feature].lower() for obj in content]
        print(f"domain: {domain_path} len: {len(texts)}")
        self.num_samples += len(texts)
        domain_inputs = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt').data
        # domain_inputs['domain'] = tensor([i] * len(content))
        domain_inputs['labels'] = tensor([self.get_label(self.labels_feature, obj) for obj in content])

        for key in domain_inputs:
            print(f"key: {key}, memory: {self.tensor_mem(domain_inputs[key])}")
        print(
            f"task: {domain_inputs['labels'].sum() / len(content)}")
        self.inputs = domain_inputs

    def get_label(self, label, obj):
        return self.labels_funcs(label)(obj[label]) if label in obj else 0

    def labels_funcs(self, label):
        if label == 'overall':
            return lambda r: int(r > 3)
        elif label == 'vote':
            return lambda v: int(int(v.replace(',', '')) > 0)

    def read_json(self, path, n_samples):
        count = 0
        json_content = []
        dont_take = -1
        task_track = 0
        with gzip.open(path, 'rb') as gzip_file:
            for line in gzip_file:  # Read one line.
                if count == n_samples:
                    break
                line = line.rstrip()
                if line:  # Any JSON data on it?
                    obj = json.loads(line)
                    if self.data_feature in obj:
                        if self.balanced:
                            if not self.get_label(self.labels_feature, obj) == dont_take:
                                json_content.append(obj)
                                count += 1
                                task_track += self.get_label(self.labels_feature, obj)
                                if 0.45 > task_track / count:
                                    dont_take = 0
                                elif 0.55 < task_track / count:
                                    dont_take = 1
                                else:
                                    dont_take = -1
                        else:
                            json_content.append(obj)
                            count += 1

            else:
                print(f"NOT ENOUGH BALANCED: {self.balanced} DATA domain:{path} count: {count}")
                raise EOFError
        return json_content

    def __getitem__(self, idx):
        item_dict = {key: val[idx] for key, val in self.inputs.items()}
        return item_dict

    def get_sample_from_domain(self, domain, idx):
        domain_inputs = self.inputs[domain]
        return {key: val[idx] for key, val in domain_inputs.items()}

    def __len__(self):
        return self.num_samples

    @staticmethod
    def tensor_mem(tens: Tensor):
        return tens.element_size() * tens.nelement()



def main():
    """
    Load the pretraining data and saves it to the path root_dir -> bookcorpus_subdataset_2.
    We chose to take 100000 samples.
    """
    t0 = time.perf_counter()
    save_bookcorpus_data = True
    train_tokenizer = False
    vocab_size = 30_522
    root_dir = '/data/user-data/niv.ko/BookCorpus'
    data_path = os.path.join(root_dir, 'bookcorpus_subdataset_2')
    tokenizer_folder_name = 'tokenizer_folder'
    tokenizer_folder = os.path.join(root_dir, tokenizer_folder_name)
    if save_bookcorpus_data:
        dataset = load_dataset('bookcorpus', split='train[10:100000]')
        data_list = []
        for sample in tqdm(dataset):
            sample = sample['text'].replace('\n', '')
            data_list.append(sample)

        with open(data_path, 'w') as data_file:
            lines = '\n'.join(data_list)
            data_file.write(lines)
    else:
        with open(data_path, 'r') as data_file:
            lines = data_file.read()

    print("done")


def main2():
    """
    split the bookcorpus dataset into train/test with ratio 0.8/0.2
    """
    root_dir = '/data/user-data/niv.ko/BookCorpus'
    data_path = os.path.join(root_dir, 'bookcorpus_subdataset_2')
    with open(data_path, 'r') as data_file:
        lines = data_file.read()
    lines = lines.split('\n')
    data_length = len(lines)
    train_ceil_idx = math.ceil(data_length * 0.8)
    train_data = lines[:train_ceil_idx]
    test_data = lines[train_ceil_idx:]
    print(f"train data amount: {len(train_data)}, test_data_amount: {len(test_data)}")
    with open(os.path.join(root_dir, 'bookcorpus_train'), 'w') as train_file:
        train_file.write(train_data)
    with open(os.path.join(root_dir, 'bookcorpus_test'), 'w') as test_file:
        test_file.write(test_data)


if __name__ == '__main__':
    main()

import os
import glob
import math
import random
import torch
from torch.utils.data import DataLoader, random_split, Sampler
from torchvision import datasets, transforms
from torchvision import transforms, datasets
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from typing import List, Optional, Sequence

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_paths, indices, split='train', feature_only=False):
        self.feature_only = feature_only
        self.feature_paths = feature_paths
        self.dataset = dataset
        if indices is None:
            indices = list(range(len(dataset)))
        self.indices = indices
        feats = []
        if feature_paths is None:
            print('Using dummy features')
            self.features = torch.zeros(len(self.dataset), 1) # (n, 1) dummy features
        else:
            for feature_path in feature_paths:
                assert os.path.exists(feature_path), f'Feature path {feature_path} does not exist'
            for feature_path in feature_paths:
                feat = torch.load(feature_path)[split]
                # assert len(self.dataset) == len(feat), f'Feature path {feature_path} has {len(feat)} entries but dataset has {len(self.dataset)}'
                feats.append(feat[indices])
            self.features = torch.cat(feats, dim=1) # (n, d)
        self.feat_dims = [feat.size(1) for feat in feats]
        self.num_features = sum(self.feat_dims)
        print(f'Feature dims: {self.feat_dims}')
        print(f'Feature dataset: {self.features.size()}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self.features[idx]
        d = self.dataset[self.indices[idx]]
        if isinstance(d, tuple):
            x, y = d
        else:
            y = d.pop('label')
            x = d
        if self.feature_only:
            return f, y
        else:
            return x, f, y


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, directory, class_file=None, num_images=None, transform=None):
        # if class_file is None:
        files = sorted(glob.glob(os.path.join(directory, "*", "*.png")))
        files = sorted(files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        if num_images is not None:
            files = files[:num_images]
        self.image_paths = files
        # self.labels = [None] * len(files)
        if class_file is not None:
            with open(class_file, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
            name_to_label = {name: i for i, name in enumerate(class_names)}
            self.labels = [name_to_label[os.path.basename(os.path.dirname(f))] for f in files]
        else:
            self.labels = [None] * len(files)
        #     class_dirs = sorted([d for d in os.listdir(directory
        # else:
        #     files = sorted(glob.glob(os.path.join(directory, "*.png")))
        #     files = sorted(files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        #     if num_images is not None:
        #         files = files[:num_images]

        #     files_dict = dict()
        #     for f in files:
        #         class_name = os.path.basename(os.path.dirname(f))
        #         if class_name not in files_dict:
        #             files_dict[class_name] = []
        #         files_dict[class_name].append(f)

        #     with open(class_file, "r") as f:
        #         class_names = [line.strip() for line in f if line.strip()]
        #     self.image_paths = []
        #     self.labels = []
        #     for label, class_name in enumerate(class_names):
        #         # class_dir = os.path.join(directory, class_name)
        #         # class_files = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        #         if num_images is None:
        #             num_class_images = len(class_files)
        #         elif isinstance(num_images, int):
        #             num_class_images = min(num_images, len(class_files))
        #         elif isinstance(num_images, dict):
        #             assert class_name in num_images, f'Class {class_name} not in num_images dict'
        #             num_class_images = min(num_images[class_name], len(class_files))
        #         self.image_paths.extend(class_files[:num_class_images])
        #         self.labels.extend([label] * num_class_images)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if label is None:
            return image, -1
        else:
            return image, label

class ConcatFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, ds1, ds2):

        assert ds1.feat_dims == ds2.feat_dims, "Feature dimensions must match"
        assert ds1.num_features == ds2.num_features, "Number of features must match"
        
        self.feat_dims = ds1.feat_dims
        self.num_features = ds1.num_features
        print(f"ConcatFeatureDataset feature dims: {self.feat_dims}")
        print(f"ConcatFeatureDataset num_features: {self.num_features}")

        self.datasets = []    
        if isinstance(ds1, FeatureDataset):
            self.datasets.append(ds1)
        elif isinstance(ds1, ConcatFeatureDataset):
            self.datasets.extend(ds1.datasets)
        else:
            raise ValueError("ds1 must be FeatureDataset or ConcatFeatureDataset")
        
        if isinstance(ds2, FeatureDataset):
            self.datasets.append(ds2)
        elif isinstance(ds2, ConcatFeatureDataset):
            self.datasets.extend(ds2.datasets)
        else:
            raise ValueError("ds2 must be FeatureDataset or ConcatFeatureDataset")

        self.lengths = [len(d) for d in self.datasets]

    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range")


class MultiFeatureDataset(torch.utils.data.Dataset):
    """Dataset that combines multiple feature datasets while preserving feature metadata."""

    def __init__(self, datasets: Sequence[torch.utils.data.Dataset]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        base_feat_dims = getattr(datasets[0], 'feat_dims', [])
        base_num_features = getattr(datasets[0], 'num_features', len(base_feat_dims))

        for ds in datasets[1:]:
            feat_dims = getattr(ds, 'feat_dims', base_feat_dims)
            num_features = getattr(ds, 'num_features', base_num_features)
            if feat_dims != base_feat_dims:
                raise ValueError("Feature dimensions must match across datasets")
            if num_features != base_num_features:
                raise ValueError("Number of features must match across datasets")

        self.datasets: List[torch.utils.data.Dataset] = list(datasets)
        self.lengths: List[int] = [len(ds) for ds in self.datasets]
        self.offsets: List[int] = []
        running_total = 0
        for length in self.lengths:
            self.offsets.append(running_total)
            running_total += length

        self.total_length = running_total
        self.feat_dims = base_feat_dims
        self.num_features = base_num_features

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for dataset, length in zip(self.datasets, self.lengths):
            if idx < length:
                return dataset[idx]
            idx -= length
        raise IndexError("Index out of range")


class FixedRatioBatchSampler(Sampler[List[int]]):
    """Batch sampler that draws a fixed ratio of samples from each dataset."""

    def __init__(
        self,
        dataset: MultiFeatureDataset,
        ratios: Sequence[float],
        batch_size: int,
        shuffle: Optional[bool] = True,
    ):
        if not isinstance(dataset, MultiFeatureDataset):
            raise TypeError("dataset must be an instance of MultiFeatureDataset")
        if len(ratios) != len(dataset.datasets):
            raise ValueError("Number of ratios must match number of datasets")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total_ratio = sum(ratios)
        if total_ratio <= 0:
            raise ValueError("Sum of ratios must be positive")

        # Normalize ratios to sum to 1 for stability
        self.ratios = [float(r) / total_ratio for r in ratios]
        self.dataset = dataset
        self.batch_size = batch_size

        self.num_batches = max([math.ceil(length / c) for length, c in zip(dataset.lengths, self._counts_for_batch(batch_size))])

        self.lengths = dataset.lengths
        self.offsets = dataset.offsets
        
        self.shuffle = shuffle

        self.indices = [list(range(length)) for length in self.lengths]     
    
    def _get_data_indices(self, dataset_index: int) -> List[int]:
        result = self.indices[dataset_index].copy()
        if self.shuffle:
            random.shuffle(result)
        return result

    def __len__(self):
        return self.num_batches

    def _counts_for_batch(self, batch_size: int) -> List[int]:
        raw_counts = [ratio * batch_size for ratio in self.ratios]
        counts = [int(math.floor(x)) for x in raw_counts]
        remainder = batch_size - sum(counts)
        if remainder > 0:
            # Distribute remaining samples based on largest fractional parts
            fractional = [(raw_counts[i] - counts[i], i) for i in range(len(raw_counts))]
            fractional.sort(reverse=True)
            for _, idx in fractional[:remainder]:
                counts[idx] += 1
        return counts

            
    def __iter__(self):
        indices = [self._get_data_indices(dataset_index) for dataset_index in range(len(self.indices))]

        counts = self._counts_for_batch(self.batch_size)

        for dataset_index in range(len(self.indices)):
            while len(indices[dataset_index]) < counts[dataset_index]:
                indices[dataset_index].extend(self._get_data_indices(dataset_index))


        max_index = len(self.dataset.datasets) - 1

        for _ in range(self.num_batches):
            batch_indices: List[int] = []
            for dataset_index, count in enumerate(counts):
                if dataset_index > max_index or count <= 0:
                    continue
                length = self.lengths[dataset_index]
                if length == 0:
                    continue

                sampled = torch.tensor(indices[dataset_index][:count])
                indices[dataset_index] = indices[dataset_index][count:]
                while len(indices[dataset_index]) < counts[dataset_index]:
                    indices[dataset_index].extend(self._get_data_indices(dataset_index))

                batch_indices.extend((sampled + self.offsets[dataset_index]).tolist())

            if batch_indices and self.shuffle:
                permutation = torch.randperm(len(batch_indices))
                batch_indices = [batch_indices[i] for i in permutation.tolist()]

            yield batch_indices

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = self.preload_dataset()
        if isinstance(dataset, torch.utils.data.Subset):
            self.indices = dataset.indices

    def preload_dataset(self):
        print('Preloading dataset...')
        cache = [None]*len(self.dataset)
        for idx in tqdm(range(len(self.dataset))):
            cache[idx] = self.dataset[idx]
        return cache

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.dataset)

def get_cifar_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

default_get_transform = {
    'cifar10': get_cifar_transform,
    'cifar100': get_cifar_transform,
}

def get_dataset(dataset, get_transform=None, tokenizer=None, no_augment=True, cache=False):
    if dataset == 'none':
        return None, None, None
    dataset = dataset.lower()
    # assert dataset in ['cifar10', 'cifar100'], f'Unknown dataset {dataset}'
    assert dataset in ['cifar10', 'cifar100', 'flowers', 'pets', 'aircraft', 'dtd', 'food',
                       'imdb', 'boolq', 'snli-ve', 'snli-ve-img', 'snli-ve-txt', 'cola', 'rte', 'mrpc', 'mnli', 'sst2', 'wnli', 'qnli', 'qqp'], f'Unknown dataset {dataset}'
    if not get_transform:
        print(f'Using default transform for {dataset}')
        get_transform = default_get_transform[dataset] if dataset in default_get_transform else lambda train: lambda x: x
    train_transform = get_transform(train=not no_augment) # no_augment overrides train transform, used for prior
    test_transform = get_transform(train=False)
    print('Train transform:')
    print(train_transform)
    print('Test transform:')
    print(test_transform)
    if dataset == 'snli-ve':
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        if train_transform is not None and test_transform is not None:
            train_ds.set_transform(train_transform)
            test_ds.set_transform(test_transform)
        else:
            # linear prob eval
            print('Assuming linear prob eval')
            # drop all but label columns
            cols = train_ds.column_names
            to_remove = [c for c in cols if c not in ['label']]
            train_ds = train_ds.remove_columns(to_remove)
            test_ds = test_ds.remove_columns(to_remove)
    elif dataset == 'snli-ve-img':
        # image only
        # train_transform: img -> img
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        train_ds.set_transform(lambda x: {'image': [train_transform(img) for img in x['image']]})
        test_ds.set_transform(lambda x: {'image': [test_transform(img) for img in x['image']]})
    elif dataset == 'snli-ve-txt':
        # text only
        assert train_transform == test_transform == None, 'train_transform and test_transform must be None for text only'
        assert tokenizer is not None, 'Must provide tokenizer'
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        train_ds.set_transform(lambda x: tokenizer(x["hypothesis"], truncation=True))
        test_ds.set_transform(lambda x: tokenizer(x["hypothesis"], truncation=True))
    elif dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root='~/data', train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root='~/data', train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train_ds = datasets.CIFAR100(root='~/data', train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR100(root='~/data', train=False, download=True, transform=test_transform)
    elif dataset == 'flowers':
        train_ds = datasets.Flowers102(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.Flowers102(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'pets':
        train_ds = datasets.OxfordIIITPet(root='~/data', split='trainval', download=True, transform=train_transform)
        test_ds = datasets.OxfordIIITPet(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'aircraft':
        train_ds = datasets.FGVCAircraft(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.FGVCAircraft(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'dtd':
        train_ds = datasets.DTD(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.DTD(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'food':
        train_ds = datasets.Food101(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.Food101(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'imdb':
        imdb = load_dataset("imdb")
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = imdb['train']
            test_ds = imdb['test']
        else:
            tokenizer.truncation_side = 'left'
            postfix = 'Overall, the sentiment of my review is'
            preprocess_function = lambda x: tokenizer([t + postfix for t in x['text']], truncation=True)
            tokenized_imdb = imdb.map(preprocess_function, batched=True)
            tokenized_imdb = tokenized_imdb.remove_columns(["text"])
            train_ds = tokenized_imdb['train']
            test_ds = tokenized_imdb['test']
    elif dataset == 'boolq':
        dataset = load_dataset('super_glue', 'boolq')
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Question: {q}\nReference: {p}\nAnswer: ' for q, p in zip(x['question'], x['passage'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question", "passage", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'mnli':
        dataset = load_dataset('glue', 'mnli') # (premise, hypothesis, label)
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation_matched']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis? Answer: ' for p, h in zip(x['premise'], x['hypothesis'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation_matched']
    elif dataset == 'mrpc':
        dataset = load_dataset('glue', 'mrpc') # (sentence1, sentence2, label (equivalent or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Sentence 1: {s1}\nSentence 2: {s2}\nIs Sentence 1 equivalent to Sentence 2? Answer: ' for s1, s2 in zip(x['sentence1'], x['sentence2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'sst2':
        dataset = load_dataset('glue', 'sst2') # (sentence, label (positive or negative))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation'] 
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Review: "{s}"\nSentiment: ' for s in x['sentence']], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'cola':
        dataset = load_dataset('glue', 'cola') # (sentence, label (sentence is grammatical or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Is the sentence "{s}" grammatical? Answer: ' for s in x['sentence']], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'qnli':
        dataset = load_dataset('glue', 'qnli') # (question, sentence, label (entailment or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'right'
            preprocess_function = lambda x: tokenizer([f'Question: {q}\nSentence: {s}\nDoes the sentence answer the question? Answer: ' for q, s in zip(x['question'], x['sentence'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question", "sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'rte':
        dataset = load_dataset('glue', 'rte') # (sentence1, sentence2, label (entailment or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Sentence 1: {s1}\nSentence 2: {s2}\nDoes Sentence 1 entail Sentence 2? Answer: ' for s1, s2 in zip(x['sentence1'], x['sentence2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'qqp':
        dataset = load_dataset('glue', 'qqp') # (question1, question2, label (equivalent or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Question 1: {q1}\nQuestion 2: {q2}\nAre Question 1 and Question 2 equivalent? Answer: ' for q1, q2 in zip(x['question1'], x['question2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question1", "question2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    return train_ds, test_ds

def split_train(train_ds, train_frac, val_frac):
    train_frac = min(train_frac, 1 - val_frac) # for explicitly subsampling train set
    train_ds, val_ds, _ = random_split(train_ds, [train_frac, val_frac, 1 - (train_frac + val_frac)], generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds

def get_loader(ds, batch_size, num_workers=0, shuffle=False, input_collate_fn=None, sampler=None, batch_sampler=None):
    if input_collate_fn is not None:
        if isinstance(ds, FeatureDataset):
            if not ds.feature_only:
                # (x, f, y) process x with input_collate_fn
                collate_fn = lambda batch: (input_collate_fn([b[0] for b in batch]), torch.stack([b[1] for b in batch]), torch.tensor([b[2] for b in batch]))
            else:
                collate_fn = None
        else:
            d = ds[0]
            if isinstance(d, tuple):
                # (x, y) process x with input_collate_fn
                collate_fn = lambda batch: (input_collate_fn([b[0] for b in batch]), torch.tensor([b[1] for b in batch]))
            else:
                # x process x with input_collate_fn
                collate_fn = lambda batch: input_collate_fn(batch)
    else:
        collate_fn = None
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'collate_fn': collate_fn,
    }

    if batch_sampler is not None:
        loader_kwargs['batch_sampler'] = batch_sampler
        # DataLoader ignores batch_size and sampler when batch_sampler provided
    else:
        loader_kwargs['batch_size'] = batch_size
        if sampler is not None:
            loader_kwargs['sampler'] = sampler
            shuffle = False
        loader_kwargs['shuffle'] = shuffle

    return DataLoader(ds, **loader_kwargs)

def get_out_dim(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'flowers':
        return 102
    elif dataset == 'pets':
        return 37
    elif dataset == 'imdb':
        return 2
    elif dataset == 'imdb':
        return 2
    elif dataset == 'cola':
        return 2
    elif dataset == 'rte':
        return 2
    elif dataset == 'boolq':
        return 2
    elif dataset == 'snli-ve':
        return 3
    elif dataset == 'sst2':
        return 2
    elif dataset == 'mrpc':
        return 2
    elif dataset == 'mnli':
        return 3
    elif dataset == 'wnli':
        return 2
    elif dataset == 'qnli':
        return 2
    elif dataset == 'qqp':
        return 2
    elif dataset == 'aircraft':
        return 100
    elif dataset == 'dtd':
        return 47
    elif dataset == 'food':
        return 101
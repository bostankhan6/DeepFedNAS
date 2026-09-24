import logging

import hashlib
import json
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pickle
from pathlib import Path
from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def _array_content_sha256(images, labels):
    digest = hashlib.sha256()
    images = np.ascontiguousarray(images)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    digest.update(str(images.shape).encode())
    digest.update(str(images.dtype).encode())
    digest.update(images.tobytes())
    digest.update(str(labels.shape).encode())
    digest.update(str(labels.dtype).encode())
    digest.update(labels.tobytes())
    return digest.hexdigest()


def _load_partition_manifest(
    manifest_path,
    X_train,
    y_train,
    partition,
    n_nets,
    alpha,
    partition_seed=None,
):
    path = Path(manifest_path).expanduser().resolve()
    with path.open() as handle:
        manifest = json.load(handle)

    expected_metadata = {
        "format_version": 1,
        "dataset": "cifar10",
        "partition_method": partition,
        "client_num_in_total": n_nets,
        "train_sample_count": len(y_train),
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Client partition manifest mismatch for {key}: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )
    if not np.isclose(float(manifest["partition_alpha"]), float(alpha)):
        raise ValueError(
            "Client partition manifest alpha does not match --partition_alpha"
        )
    if partition_seed is not None and manifest.get("partition_seed") != partition_seed:
        raise ValueError(
            "Client partition manifest seed does not match "
            "--client_partition_seed"
        )

    data_hash = _array_content_sha256(X_train, y_train)
    if manifest.get("train_data_sha256") != data_hash:
        raise ValueError(
            "Client partition manifest was generated from a different clean "
            "training split"
        )

    manifest_clients = manifest.get("client_indices", {})
    if set(manifest_clients) != {str(index) for index in range(n_nets)}:
        raise ValueError("Client partition manifest has unexpected client keys")
    net_dataidx_map = {}
    for client_index in range(n_nets):
        recorded = np.asarray(manifest_clients[str(client_index)], dtype=np.int64)
        net_dataidx_map[client_index] = recorded

    all_indices = np.concatenate(list(net_dataidx_map.values()))
    if len(all_indices) != len(y_train):
        raise ValueError("Client partition manifest does not cover every sample")
    if not np.array_equal(np.sort(all_indices), np.arange(len(y_train))):
        raise ValueError(
            "Client partition manifest must cover each training sample exactly once"
        )
    partition_bytes = json.dumps(
        {
            str(client): [int(index) for index in net_dataidx_map[client]]
            for client in range(n_nets)
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        manifest.get("partition_indices_sha256")
        != hashlib.sha256(partition_bytes).hexdigest()
    ):
        raise ValueError("Client partition index hash mismatch")

    logging.info("Loaded locked client partition manifest: %s", path)
    return net_dataidx_map


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(
    augmentation="basic", randaugment_num_ops=2, randaugment_magnitude=6,
):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if augmentation not in ("basic", "mixaug"):
        raise ValueError(
            f"Unsupported CIFAR-10 augmentation {augmentation!r}; "
            "expected 'basic' or 'mixaug'."
        )

    train_transforms = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "mixaug":
        train_transforms.append(
            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            )
        )
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if augmentation == "basic":
        train_transforms.append(Cutout(16))
    train_transform = transforms.Compose(train_transforms)

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _load_pickle_split(datadir, split_name):
    with open(f'{datadir}/{split_name}.pkl', 'rb') as file:
        split_data, split_target = pickle.load(file)
    return np.asarray(split_data), np.asarray(split_target)


def _load_clean_cifar10_splits(datadir):
    train_data, train_target = _load_pickle_split(datadir, 'train')
    validation_data, validation_target = _load_pickle_split(datadir, 'val')
    if len(train_data) == len(validation_data) and np.array_equal(train_data, validation_data):
        raise ValueError(
            'The CIFAR-10 train.pkl and val.pkl files are identical; '
            'regenerate a valid clean split before training.'
        )
    return train_data, train_target, validation_data, validation_target


def load_cifar10_data(
    datadir, load_from_pkl=False, augmentation="basic",
    randaugment_num_ops=2, randaugment_magnitude=6,
):
    train_transform, test_transform = _data_transforms_cifar10(
        augmentation, randaugment_num_ops, randaugment_magnitude,
    )

    if load_from_pkl:
        X_train, y_train, X_validation, y_validation = _load_clean_cifar10_splits(datadir)
        return X_train, y_train, X_validation, y_validation

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(
    dataset,
    datadir,
    partition,
    n_nets,
    alpha,
    load_from_pkl=False,
    client_partition_manifest=None,
    client_partition_seed=None,
):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir, load_from_pkl=load_from_pkl)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if client_partition_manifest is not None:
        net_dataidx_map = _load_partition_manifest(
            client_partition_manifest,
            X_train,
            y_train,
            partition,
            n_nets,
            alpha,
            client_partition_seed,
        )

    elif partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(
    dataset, datadir, train_bs, test_bs, dataidxs=None, load_from_pkl=False,
    augmentation="basic",
    randaugment_num_ops=2, randaugment_magnitude=6,
):
    return get_dataloader_CIFAR10(
        datadir, train_bs, test_bs, dataidxs, load_from_pkl, augmentation,
        randaugment_num_ops, randaugment_magnitude,
    )


# for local devices
def get_dataloader_test(
    dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test,
    augmentation="basic",
    randaugment_num_ops=2, randaugment_magnitude=6,
):
    return get_dataloader_test_CIFAR10(
        datadir, train_bs, test_bs, dataidxs_train, dataidxs_test, augmentation,
        randaugment_num_ops, randaugment_magnitude,
    )


def get_dataloader_CIFAR10(
    datadir, train_bs, test_bs, dataidxs=None, load_from_pkl=False,
    augmentation="basic",
    randaugment_num_ops=2, randaugment_magnitude=6,
):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10(
        augmentation, randaugment_num_ops, randaugment_magnitude,
    )

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    if load_from_pkl:
        train_data, train_target, validation_data, validation_target = _load_clean_cifar10_splits(datadir)
        if dataidxs is not None:
            train_data, train_target = train_data[dataidxs], train_target[dataidxs]
        train_ds.data, train_ds.target = train_data, train_target
        validation_ds = dl_obj(datadir, train=True, transform=transform_test, download=True)
        validation_ds.data, validation_ds.target = validation_data, validation_target
    else:
        validation_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    validation_dl = data.DataLoader(dataset=validation_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    train_dl.split_name = 'train'
    validation_dl.split_name = 'validation' if load_from_pkl else 'test'

    return train_dl, validation_dl


def get_dataloader_test_CIFAR10(
    datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None,
    augmentation="basic",
    randaugment_num_ops=2, randaugment_magnitude=6,
):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10(
        augmentation, randaugment_num_ops, randaugment_magnitude,
    )

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    val_batch_size=256,
    load_from_pkl=False,
    client_partition_manifest=None,
    client_partition_seed=None,
    augmentation="basic",
    randaugment_num_ops=2,
    randaugment_magnitude=6,
):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha,
                                                                                             load_from_pkl,
                                                                                             client_partition_manifest,
                                                                                             client_partition_seed)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, val_batch_size,
        load_from_pkl=load_from_pkl, augmentation=augmentation,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, batch_size, val_batch_size, dataidxs,
            load_from_pkl, augmentation, randaugment_num_ops,
            randaugment_magnitude,
        )
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

import numpy as np
import os
import random
import scipy.sparse
import urllib.request

from collections import namedtuple
from zipfile import ZipFile

class DDData(object):
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"
    filenames = [
        "DD_A.txt", "DD_graph_indicator.txt", "DD_graph_labels.txt",
        "DD_node_labels.txt"
    ]

    def __init__(self, data_root="data", split_ratio=[0.4, 0.3, 0.3]):
        self.data_root = data_root
        self.maybe_download()
        adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        self.adjacency = adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        self.train_index, self.val_index, self.test_index = \
            self.split_data(split_ratio)

    def read_data(self):
        data_dir = os.path.join(self.data_root, "DD")
        print("loading DD_A.txt")
        adjacency = np.genfromtxt(
            fname=os.path.join(data_dir, "DD_A.txt"),
            dtype=np.int64,
            delimiter=","
        ) - 1
        print("loading DD_node_labels.txt")
        node_labels = np.genfromtxt(
            fname=os.path.join(data_dir, "DD_node_labels.txt"),
            dtype=np.int64
        ) - 1
        print("loading DD_graph_indicator.txt")
        graph_indicator = np.genfromtxt(
            fname=os.path.join(data_dir, "DD_graph_indicator.txt"),
            dtype=np.int64
        ) - 1
        print("loading DD_graph_labels.txt")
        graph_labels = np.genfromtxt(
            fname=os.path.join(data_dir, "DD_graph_labels.txt"),
            dtype=np.int64
        ) - 1
        num_nodes = len(node_labels)
        adjacency = scipy.sparse.coo_matrix(
            (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        print("number of nodes: ", num_nodes)
        return adjacency, node_labels, graph_indicator, graph_labels

    def split_data(self, split_ratio):
        train_index, val_index, test_index = [],  [], []
        splits = [split_ratio[0], sum(split_ratio[0 : 2])]
        for idx in range(len(self.graph_labels)):
            split = random.random()
            if split < splits[0]:
                train_index.append(idx)
            elif split < splits[1]:
                val_index.append(idx)
            else:
                test_index.append(idx)
        return train_index, val_index, test_index

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "DD")
        for name in self.filenames:
            if not os.path.isfile(os.path.join(save_path, name)):
                self.download_data(
                    self.url, self.data_root
                )
                break

    @staticmethod
    def download_data(url, save_path):
        """原始数据不存在时，下载数据"""
        print("downloading data from {}".format(url))
        filename = os.path.basename(url)
        filepath = os.path.join(save_path, filename)
        if not os.path.isfile(filepath):
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            data = urllib.request.urlopen(url)
            with open(filepath, mode="wb") as f:
                f.write(data.read())
        # unzip
        with ZipFile(filepath, "r") as zfile:
            zfile.extractall(save_path)
            print("Extracting data from {}".format(filepath))
        return True

    def __getitem__(self, index):
        """index: 图索引"""
        # 节点掩膜
        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_label = self.graph_labels[index]
        adjacency = self.adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_label

    def __len__(self):
        return len(self.graph_labels)

if __name__ == "__main__":
    data_root = r"D:\proj\gnn\data"
    dd = DDData(data_root)
    sample = dd[0]
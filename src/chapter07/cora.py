import numpy as np
import os
import pickle
import random
import tarfile
import gzip
import urllib

from collections import namedtuple
from scipy.sparse import coo_matrix, diags, eye

Data = namedtuple(
    "Data", [
        "x", "y", "adjacency", "graph", "train_mask", "val_mask", "test_mask",
        "id2vid", "id2cls"
    ]
)

class CoraData(object):
    download_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    filenames = ["cora.cites", "cora.content"]
    def __init__(
        self, data_root="data", rebuild=False, split_ratio=[0.4, 0.3, 0.3]
    ):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/cora
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据
            split_ratio: list, optional
                train, dev, test比例
        """
        self.data_root = data_root
        self.split_ratio = split_ratio
        save_file = os.path.join(data_root, "processed_cora.pkl")
        if os.path.isfile(save_file) and not rebuild:
            print("using cached file: {}".format(save_file))
            with open(save_file, mode="rb") as f:
                self._data = pickle.load(f)
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, mode="wb") as f:
                pickle.dump(self.data, f)
            print("cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象
        包括x, y, adjacency, graph, train_mask, val_mask, test_mask, id2vid, id2cls
        """
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        """
        print("process data ...")
        vetices, edges, classes = self.read_data(
            os.path.join(self.data_root, "cora")
        )
        x, y, edges, id2vid, id2cls = self.to_numpy(
            vetices, edges, classes
        )
        num_nodes = x.shape[0]
        # split train, val and test sets using mask
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        splits = [self.split_ratio[0], sum(self.split_ratio[0 : 2])]
        for idx in range(num_nodes):
            split = random.random()
            if split < splits[0]:
                train_mask[idx] = True
            elif split < splits[1]:
                val_mask[idx] = True
            else:
                test_mask[idx] = True
        adjacency = self.build_adjacency(edges, num_nodes)
        graph = self.build_graph(edges, num_nodes)
        print("node's feature shape: ", x.shape)
        print("node's label shape: ", y.shape)
        print("adjacency's shape: ", adjacency.shape)
        print("number of training nodes: ", train_mask.sum())
        print("number of validation nodes: ", val_mask.sum())
        print("number of test nodes: ", test_mask.sum())
        return Data(
            x=x, y=y, adjacency=adjacency, graph=graph,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
            id2vid=id2vid, id2cls=id2cls
        )

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "cora")
        for name in self.filenames:
            if not os.path.isfile(os.path.join(save_path, name)):
                self.download_data(
                    self.download_url, self.data_root
                )
                break

    @staticmethod
    def download_data(url, root):
        """原始数据不存在时，下载数据"""
        filename = os.path.basename(url)
        filepath = os.path.join(root, filename)
        if not os.path.isfile(filepath):
            if not os.path.isdir(root):
                os.makedirs(root)
            data = urllib.request.urlopen(url)
            with open(filepath, mode="wb") as f:
                f.write(data.read())
        # unzip
        gfile = gzip.GzipFile(filepath)
        filepath = filepath.replace("tgz", "tar")
        with open(filepath, mode = "wb") as f:
            f.write(gfile.read())
        gfile.close()
        # extract
        with tarfile.open(filepath) as tfile:
            tfile.extractall(root)
        os.remove(filepath)
        return True

    @staticmethod
    def read_data(root):
        """读取原始数据"""
        filepath = os.path.join(root, "cora.content")
        vetices = {}
        classes = set()
        with open(filepath, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            vid = line[0]
            one_hot = [int(item) for item in line[1 : -1]]
            label = line[-1]
            vetices[vid] = [one_hot, label]
            classes.add(label)
        filepath = os.path.join(root, "cora.cites")
        edges = {}
        with open(filepath, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            src, dst = line.strip().split("\t")
            edges.setdefault(src, [])
            edges[src].append(dst)
        return vetices, edges, classes

    @staticmethod
    def to_numpy(vetices, edges, classes):
        vid2id = {vid: idx for idx, vid in enumerate(vetices)}
        cls2id = {c: idx for idx, c in enumerate(classes)}
        id2vid = {idx: vid for idx, vid in enumerate(vetices)}
        id2cls = {idx: c for idx, c in enumerate(classes)}
        x = np.asarray(
            [vetices[key][0] for key in vetices]
        )
        y = np.asarray(
            [cls2id[vetices[key][1]] for key in vetices]
        )
        edges = {
            vid2id[src]: [vid2id[dst] for dst in edges[src]] for src in edges
        }
        return x, y, edges, id2vid, id2cls

    @staticmethod
    def build_adjacency(edges, num_nodes):
        edge_set = set()
        for src in edges:
            edge_set.update((src, dst) for dst in edges[src])
            edge_set.update((dst, src) for dst in edges[src])
        edge_set = np.asarray(list(edge_set))
        adjacency = coo_matrix(
            (np.ones(edge_set.shape[0]), (edge_set[:, 0], edge_set[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        return adjacency

    @staticmethod
    def build_graph(edges, num_nodes):
        graph = {vid: set() for vid in range(num_nodes)}
        for src in edges:
            for dst in edges[src]:
                graph[src].add(dst)
                graph[dst].add(src)
        return {vid: list(neighbors) for vid, neighbors in graph.items()}

    @staticmethod
    def normalization(adjacency):
        """计算 L = D^-0.5 * (A+I) * D^-0.5"""
        # 增加自连接
        adjacency += eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1)).flatten()
        d_hat = diags(np.power(degree, -0.5))
        return d_hat.dot(adjacency).dot(d_hat).tocoo()

if __name__ == "__main__":
    data_root = "./data"
    cora = CoraData(data_root)
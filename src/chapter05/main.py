import numpy as np
import torch
from torch import nn
from torch import optim

from cora import CoraData
from gcn import GcnNet
from train import train, test, tensor_from_numpy
from visual import plot_loss_with_acc, plot_tsne

if __name__ == "__main__":
    # 参数设置
    LEARNING_RATE = 0.1
    WEIGHT_DACAY = 5e-4
    EPOCHS = 200
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练数据
    data_root = r"D:\proj\gnn\data"
    dataset = CoraData(data_root).data
    # 归一化数据，使得每一行和为1
    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)
    num_nodes, input_dim = node_feature.shape
    num_classes = len(dataset.id2cls)
    x = tensor_from_numpy(node_feature, DEVICE).float()
    y = tensor_from_numpy(dataset.y, DEVICE).long()
    train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
    val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
    test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
    # 归一化邻接矩阵
    normalize_adjacency = CoraData.normalization(dataset.adjacency)
    indices = torch.from_numpy(np.asarray([
        normalize_adjacency.row, normalize_adjacency.col
    ])).long()
    values = torch.from_numpy(normalize_adjacency.data).float()
    adjacency = torch.sparse.FloatTensor(
        indices, values, (num_nodes, num_nodes)
    ).to(DEVICE)

    # 模型
    model = GcnNet(input_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(
        params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY
    )

    # 训练
    loss, val_acc = train(
        model, x, y, adjacency, train_mask, val_mask,
        criterion, optimizer, EPOCHS
    )
    plot_loss_with_acc(loss, val_acc)

    # 测试
    test_acc, test_logits, test_label = test(model, x, y, adjacency, test_mask)
    print("test accuracy: {:.4}".format(test_acc.item()))
    # 绘制测试数据的TSNE降维图
    plot_tsne(test_logits, test_label, dataset.id2cls)
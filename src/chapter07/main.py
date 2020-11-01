import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cora import CoraData
from net import GraphSage
from train import train, test
from visual import plot_loss_with_acc, plot_tsne

if __name__ == "__main__":
    # 输入维度
    INPUT_DIM = 1433
    # 采样邻居阶数与GCN层数一致
    # 隐藏单元节点数
    HIDDEN_DIM = [128, 7]
    # 每阶采样邻居的节点数
    NUM_NEIGHBORS_LIST = [10, 10]
    assert len(NUM_NEIGHBORS_LIST) == len(HIDDEN_DIM)
    # 批处理大小
    BATCH_SIZE = 16
    EPOCHS = 20
    # 每个epoch循环的批次数
    NUM_BATCH_PER_EPOCH = 20
    # 学习率
    LEARNING_RATE = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练数据
    data_root = r"D:\proj\gnn\data"
    data = CoraData(data_root, rebuild=False).data
    # 归一化数据，使得每一行和为1
    x = data.x / data.x.sum(1, keepdims=True)
    train_index = np.where(data.train_mask)[0]
    val_index = np.where(data.val_mask)[0]
    test_index = np.where(data.test_mask)[0]

    # 模型
    model = GraphSage(INPUT_DIM, HIDDEN_DIM, NUM_NEIGHBORS_LIST)
    print(model)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
    )
    # 训练
    loss_history, val_acc_history = train(
        model, x, data.y, data.graph, train_index, val_index, NUM_NEIGHBORS_LIST,
        criterion, optimizer, EPOCHS, NUM_BATCH_PER_EPOCH, BATCH_SIZE, DEVICE
    )
    plot_loss_with_acc(loss_history, val_acc_history)
    # 测试
    test_acc, test_logits, test_label = test(
        model, x, data.y, data.graph, test_index, NUM_NEIGHBORS_LIST, BATCH_SIZE, DEVICE
    )
    plot_tsne(test_logits, test_label, data.id2cls)
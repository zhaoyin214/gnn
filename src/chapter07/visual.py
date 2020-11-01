import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(
        range(len(loss_history)),
        loss_history,
        c=np.array([255, 71, 90]) / 255
    )
    plt.ylabel("loss")
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(
        range(len(val_acc_history)),
        val_acc_history,
        c=np.array([79, 179, 255]) / 255
    )
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("val acc")
    plt.xlabel("epoch")
    plt.title("train loss & validation accuracy")
    plt.show()

def plot_tsne(logits, label, id2cls):
    tsne = TSNE()
    out = tsne.fit_transform(logits)
    fig = plt.figure()
    for i in range(len(id2cls)):
        indices = label == i
        x, y = out[indices].T
        plt.scatter(x, y, label=id2cls[i])
    plt.legend()
    plt.show()
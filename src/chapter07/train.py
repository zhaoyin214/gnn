import numpy as np
import torch
from sampling import multihop_sampling

def train(
    model, x, y, graph, train_index, val_index, num_neighbors_list,
    criterion, optimizer, epochs, num_batch_per_epoch, batch_size, device
):
    model.to(device)
    loss_history = []
    val_acc_history = []
    for epoch in range(epochs):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        for batch in range(num_batch_per_epoch):
            # 前向传播
            def batch_forward(index):
                batch_src_index = np.random.choice(index, size=(batch_size, ))
                batch_src_label = torch.from_numpy(
                    y[batch_src_index]
                ).long().to(device)
                batch_sampling_result = multihop_sampling(
                    batch_src_index, num_neighbors_list, graph
                )
                batch_sampling_x = [
                    torch.from_numpy(x[idx]).float().to(device)
                    for idx in batch_sampling_result
                ]
                batch_logits = model(batch_sampling_x)
                return batch_logits, batch_src_label

            # 训练
            model.train()
            optimizer.zero_grad()
            batch_logits, batch_src_label = batch_forward(train_index)
            loss = criterion(batch_logits, batch_src_label)
            # 反向传播计算参数的梯度
            loss.backward()
            # 使用优化方法进行梯度更新
            optimizer.step()
            train_loss += loss.item()
            train_acc += evaluate(batch_logits, batch_src_label)

            # 验证
            model.eval()
            batch_logits, batch_src_label = batch_forward(val_index)
            val_acc += evaluate(batch_logits, batch_src_label)
            print(
                "epoch {:03d} batch {:03d}: loss {:.4f}".format(
                    epoch, batch, loss.item()
                )
            )
        train_loss /= num_batch_per_epoch
        train_acc /= num_batch_per_epoch
        val_acc /= num_batch_per_epoch
        print(
            "epoch {:03d}: loss {:.4f}, train acc {:.4}, val acc {:.4f}".format(
                epoch, train_loss, train_acc, val_acc
                )
        )
        loss_history.append(train_loss)
        val_acc_history.append(val_acc)
    return loss_history, val_acc_history

def test(
    model, x, y, graph, test_index, num_neighbors_list, batch_size, device
):
    model.eval()
    test_acc = 0
    num_batches = int(np.ceil(len(test_index) / batch_size))
    test_logits = []
    test_label = []
    with torch.no_grad():
        for batch in range(num_batches):
            # 前向传播
            def batch_forward(index):
                batch_src_index = np.random.choice(index, size=(batch_size, ))
                batch_src_label = torch.from_numpy(
                    y[batch_src_index]
                ).long().to(device)
                batch_sampling_result = multihop_sampling(
                    batch_src_index, num_neighbors_list, graph
                )
                batch_sampling_x = [
                    torch.from_numpy(x[idx]).float().to(device)
                    for idx in batch_sampling_result
                ]
                batch_logits = model(batch_sampling_x)
                return batch_logits, batch_src_label

            batch_logits, batch_src_label = batch_forward(test_index)
            test_acc += evaluate(batch_logits, batch_src_label)
            test_logits.append(batch_logits.cpu().numpy())
            test_label.append(batch_src_label.cpu().numpy())
    test_acc /= num_batches
    print("test acc {:.4}".format(test_acc))
    test_logits = np.vstack(test_logits)
    test_label = np.hstack(test_label)
    return test_acc, test_logits, test_label

def evaluate(logits, y):
    return torch.eq(logits.max(1)[1], y).float().mean()
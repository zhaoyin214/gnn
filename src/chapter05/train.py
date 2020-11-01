import torch

def tensor_from_numpy(x, device="cpu"):
    return torch.from_numpy(x).to(device)

def train(
    model, x, y, adjacency, train_mask, val_mask,
    criterion, optimizer, epochs
):
    loss_history = []
    val_acc_history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 前向传播
        logits = model(adjacency, x)
         # 计算训练节点损失
        loss = criterion(logits[train_mask], y[train_mask])
        # 反向传播，计算参数的梯度
        loss.backward()
        # 梯度更新
        optimizer.step()
        # 训练集准确率
        train_acc = evaluate(logits, y, train_mask)
        # 验证集准确率
        val_acc = evaluate(logits, y, val_mask)
        # 记录训练损失、准确率
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print(
            "epoch {:03d}: loss {:.4f}, train acc {:.4}, val acc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_acc.item()
                )
        )
    return loss_history, val_acc_history

def test(model, x, y, adjacency, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(adjacency, x)
        acc = evaluate(logits, y, test_mask)
    return acc, logits[test_mask].cpu().numpy(), y[test_mask].cpu().numpy()

def evaluate(logits, y, mask):
    pred_y = logits[mask].max(1)[1]
    acc = torch.eq(pred_y, y[mask]).float().mean()
    return acc
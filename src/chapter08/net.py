import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.init_param()

    def init_param(self):
        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + " (" \
            + str(self.input_dim) + " -> " \
            + str(self.output_dim) + ")"

class SelfAttnPool(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        """
        Arguments:
        ----------
            keep_ratio: float
                要保留的节点比例，保留的节点数量为int(N * keep_ratio)
        """
        super(SelfAttnPool, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConv(input_dim, 1, use_bias=False)

    def forward(self, adjacency, input_feature, graph_indicator):
        score = self.attn_gcn(adjacency, input_feature)
        score = self.activation(score)

        mask = self.top_rank(score, graph_indicator)

    def top_rank(self, score, graph_indicator):
        """基于给定的attention_score，对每个图进行pooling操作。
        将每个图单独进行池化，最后再将它们级联起来进行下一步计算

        Arguments:
        ----------
            score: torch.Tensor
                使用GCN计算的注意力分数，Z = GCN(A, X)
            graph_indicator: torch.Tensor
                指示每个节点属于哪个图
        """
        pass
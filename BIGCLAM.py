import numpy as np
import networkx as nx


def sigm(x):
    # sigmoid操作 求梯度会用到
    # numpy.divide数组对应位置元素做除法。
    return np.divide(np.exp(-1. * x), 1. - np.exp(-1. * x))


def log_likelihood(F, A):
    # 代入计算公式计算log似然度
    A_soft = F.dot(F.T)

    # 用邻接矩阵可以帮助我们只取到相邻的两个节点
    FIRST_PART = A * np.log(1. - np.exp(-1. * A_soft))
    sum_edges = np.sum(FIRST_PART)
    # 1-A取的不相邻的节点
    SECOND_PART = (1 - A) * A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli


def gradient(F, A, i):
    # 代入公式计算梯度值
    N, C = F.shape

    # 通过邻接矩阵找到相邻 和 不相邻节点
    neighbours = np.where(A[i])
    nneighbours = np.where(1 - A[i])

    # 公式第一部分
    sum_neigh = np.zeros((C,))
    for nb in neighbours[0]:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb] * sigm(dotproduct)

    # 公式第二部分
    sum_nneigh = np.zeros((C,))
    # Speed up this computation using eq.4
    for nnb in nneighbours[0]:
        sum_nneigh += F[nnb]

    grad = sum_neigh - sum_nneigh
    return grad


def train(A, C, iterations=100):
    # 初始化F
    N = A.shape[0]
    F = np.random.rand(N, C)

    # 梯度下降最优化F
    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)

            F[person] += 0.005 * grad

            F[person] = np.maximum(0.001, F[person])  # F应该大于0
        ll = log_likelihood(F, A)
        print('At step %5i/%5i ll is %5.3f' % (n, iterations, ll))
    return F


# 加载图数据集
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


if __name__ == "__main__":
    # adj = np.load('data/adj.npy')
    G = load_graph('data/club.txt')
    # adj = np.array(nx.adjacency_matrix(G).todense())
    adj = nx.to_numpy_array(G)  # 邻接矩阵

    F = train(adj, 4)
    F_argmax = np.argmax(F, 1)

    for i, row in enumerate(F):
        print(row)

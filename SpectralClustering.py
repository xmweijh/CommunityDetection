import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg as linalg
from matplotlib import pyplot as plt


def partition(G, k):
    # 获得邻接矩阵
    A = nx.to_numpy_array(G)
    # 获得度矩阵
    D = degree_matrix(G)
    # 获得拉普拉斯算子
    L = D - A

    # 获得归一化拉普拉斯算子Lsm
    Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
    L = np.dot(np.dot(Dn, L), Dn)
    # L = np.dot(Dn, L)

    # 获得特征值，特征向量
    eigvals, eigvecs = linalg.eig(L)
    n = len(eigvals)


    dict_eigvals = dict(zip(eigvals, range(0, n)))

    # 获得前k个特征值
    k_eigvals = np.sort(eigvals)[0:k]
    eigval_indexs = [dict_eigvals[k] for k in k_eigvals]
    k_eigvecs = eigvecs[:, eigval_indexs]

    # 归一化
    # sum_co = k_eigvecs.sum(axis=0)
    # norm_ans = k_eigvecs/sum_co

    # 使用k-means聚类
    result = KMeans(n_clusters=k).fit_predict(k_eigvecs)
    # result = KMeans(n_clusters=k).fit_predict(norm_ans)
    return result



def degree_matrix(G):
    n = G.number_of_nodes()
    V = [node for node in G.nodes()]
    D = np.zeros((n, n))
    for i in range(n):
        node = V[i]
        d_node = G.degree(node)
        D[i][i] = d_node
    return np.array(D)


if __name__ == '__main__':

    G = nx.read_edgelist("data/football.txt")
    k = 12
    sc_com = partition(G, k)
    print(sc_com)

    # 可视化
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=70, width=0.5, node_color=sc_com)
    plt.show()

    V = [node for node in G.nodes()]
    com_dict = {node: com for node, com in zip(V, sc_com)}
    com = [[V[i] for i in range(G.number_of_nodes()) if sc_com[i] == j] for j in range(k)]

    # 构造可视化所需要的图
    G_graph = nx.Graph()
    for each in com:
        G_graph.update(nx.subgraph(G, each))  #
    color = [com_dict[node] for node in G_graph.nodes()]

    # 可视化
    pos = nx.spring_layout(G_graph, seed=4, k=0.33)
    nx.draw(G, pos, with_labels=False, node_size=1, width=0.1, alpha=0.2)
    nx.draw(G_graph, pos, with_labels=True, node_color=color, node_size=70, width=0.5, font_size=5,
            font_color='#000000')
    plt.show()
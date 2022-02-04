import collections
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


class COPRA:
    def __init__(self, G, T, v):
        """
        :param G:图本身
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._v = v

    def execute(self):
        # 建立成员标签记录
        # 节点将被分配隶属度大于阈值的社区标签
        lablelist = {i: {i: 1} for i in self._G.nodes()}
        for t in range(self._T):
            visitlist = list(self._G.nodes())
            # 随机排列遍历顺序
            np.random.shuffle(visitlist)
            # 开始遍历节点
            for visit in visitlist:
                temp_count = 0
                temp_label = {}
                total = len(self._G[visit])
                # 根据邻居利用公式计算标签
                for i in self._G.neighbors(visit):
                    res = {key: value / total for key, value in lablelist[i].items()}
                    temp_label = dict(Counter(res) + Counter(temp_label))
                temp_count = len(temp_label)
                temp_label2 = temp_label.copy()
                for key, value in list(temp_label.items()):
                    if value < 1 / self._v:
                        del temp_label[key]
                        temp_count -= 1
                # 如果一个节点中所有的标签都低于阈值就随机选择一个
                if temp_count == 0:
                    # temp_label = {}
                    # v = self._v
                    # if self._v > len(temp_label2):
                    #     v = len(temp_label2)
                    # b = random.sample(temp_label2.keys(), v)
                    # tsum = 0.0
                    # for i in b:
                    #     tsum += temp_label2[i]
                    # temp_label = {i: temp_label2[i]/tsum for i in b}
                    b = random.sample(temp_label2.keys(), 1)
                    temp_label = {b[0]: 1}
                # 否则标签个数一定小于等于v个 进行归一化即可
                else:
                    tsum = sum(temp_label.values())
                    temp_label = {key: value / tsum for key, value in temp_label.items()}
                lablelist[visit] = temp_label

        communities = collections.defaultdict(lambda: list())
        # 扫描lablelist中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in lablelist.items():
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        return communities.values()


def cal_EQ(cover, G):
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # 存储每个节点所在的社区
    vertex_community = collections.defaultdict(lambda: set())
    # i为社区编号(第几个社区) c为该社区中拥有的节点
    for i, c in enumerate(cover):
        # v为社区中的某一个节点
        for v in c:
            # 根据节点v统计他所在的社区i有哪些
            vertex_community[v].add(i)
    total = 0.0
    for c in cover:
        for i in c:
            # o_i表示i节点所同时属于的社区数目
            o_i = len(vertex_community[i])
            # k_i表示i节点的度数(所关联的边数)
            k_i = len(G[i])
            for j in c:
                t = 0.0
                # o_j表示j节点所同时属于的社区数目
                o_j = len(vertex_community[j])
                # k_j表示j节点的度数(所关联的边数)
                k_j = len(G[j])
                if G.has_edge(i, j):
                    t += 1.0 / (o_i * o_j)
                t -= k_i * k_j / (2 * m * o_i * o_j)
                total += t
    return round(total / (2 * m), 4)


def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


if __name__ == '__main__':
    # G = nx.karate_club_graph()
    G = load_graph('data/dolphin.txt')
    algorithm = COPRA(G, 20, 3)

    communities = algorithm.execute()
    for i, community in enumerate(communities):
        print(i, community)

    print(cal_EQ(communities, G))
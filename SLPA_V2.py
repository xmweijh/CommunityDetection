import collections
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SLPA:
    def __init__(self, G, T, r):
        """
        :param G:图本身
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._r = r

    def execute(self):
        # 节点存储器初始化
        node_memory = []
        for i in range(self._n):
            node_memory.append({i: 1})

        # 算法迭代过程
        for t in range(self._T):
            # 任意选择一个监听器
            # np.random.permutation()：随机排列序列
            order = [x for x in np.random.permutation(self._n)]
            for i in order:
                label_list = {}
                # 从speaker中选择一个标签传播到listener
                for j in self._G.neighbors(i):
                    sum_label = sum(node_memory[j].values())
                    label = list(node_memory[j].keys())[np.random.multinomial(
                        1, [float(c) / sum_label for c in node_memory[j].values()]).argmax()]
                    label_list[label] = label_list.setdefault(label, 0) + 1
                # listener选择一个最流行的标签添加到内存中
                max_v = max(label_list.values())
                # selected_label = max(label_list, key=label_list.get)
                selected_label = random.choice([item[0] for item in label_list.items() if item[1] == max_v])
                # setdefault如果键不存在于字典中，将会添加键并将值设为默认值。
                node_memory[i][selected_label] = node_memory[i].setdefault(selected_label, 0) + 1

        # 根据阈值threshold删除不符合条件的标签
        for memory in node_memory:
            sum_label = sum(memory.values())
            threshold_num = sum_label * self._r
            for k, v in list(memory.items()):
                if v < threshold_num:
                    del memory[k]

        communities = collections.defaultdict(lambda: list())
        # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in enumerate(node_memory):
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        return communities.values()


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


# 可视化划分结果
def showCommunity(G, partition, pos):
    # 划分在同一个社区的用一个符号表示，不同社区之间的边用黑色粗体
    cluster = {}
    labels = {}
    for index, item in enumerate(partition):
        for nodeID in item:
            labels[nodeID] = r'$' + str(nodeID) + '$'  # 设置可视化label
            cluster[nodeID] = index  # 节点分区号

    # 可视化节点
    colors = ['r', 'g', 'b', 'y', 'm']
    shapes = ['v', 'D', 'o', '^', '<']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape=shapes[index],
                               node_size=350,
                               alpha=1)

    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的link
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的link
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)

    for index, edgelist in enumerate(edges.values()):
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        else:
            # cluster间
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=3, alpha=0.8, edge_color=colors[index])

    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.axis('off')
    plt.show()


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
    # pos = nx.spring_layout(G)
    G = load_graph('data/dolphin.txt')
    algorithm = SLPA(G, 20, 0.5)
    communities = algorithm.execute()
    for i, community in enumerate(communities):
        print(i, community)

    print(cal_EQ(communities, G))
import collections
import random
import time
import networkx as nx
import matplotlib.pyplot as plt


class LPA:
    def __init__(self, G, max_iter=20):
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._max_iter = max_iter

    # 判断是否收敛
    def can_stop(self):
        # 每个节点的标签和邻近节点最多的标签一样
        for i in range(self._n):
            node = self._G.nodes[i]
            label = node["label"]
            max_labels = self.get_max_neighbor_label(i)
            if label not in max_labels:
                return False
        return True

    # 获得邻近节点最多的标签
    def get_max_neighbor_label(self, node_index):
        m = collections.defaultdict(int)
        for neighbor_index in self._G.neighbors(node_index):
            neighbor_label = self._G.nodes[neighbor_index]["label"]
            m[neighbor_label] += 1
        max_v = max(m.values())
        # 可能多个标签数目相同，这里都要返回
        return [item[0] for item in m.items() if item[1] == max_v]

    # 异步更新
    def populate_label(self):
        # 随机访问
        visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
        for i in visitSequence:
            node = self._G.nodes[i]
            label = node["label"]
            max_labels = self.get_max_neighbor_label(i)
            # 如果标签不在最大标签集中才更新，否则相同随机选取没有意义
            if label not in max_labels:
                newLabel = random.choice(max_labels)
                node["label"] = newLabel

    # 根据标签得到社区结构
    def get_communities(self):
        communities = collections.defaultdict(lambda: list())
        for node in self._G.nodes(True):
            label = node[1]["label"]
            communities[label].append(node[0])
        return communities.values()

    def execute(self):
        # 初始化标签
        for i in range(self._n):
            self._G.nodes[i]["label"] = i
        iter_time = 0
        # 更新标签
        while (not self.can_stop() and iter_time < self._max_iter):
            self.populate_label()
            iter_time += 1
        return self.get_communities()


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


if __name__ == '__main__':
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    start_time = time.time()
    algorithm = LPA(G)
    communities = algorithm.execute()
    end_time = time.time()
    for community in communities:
        print(community)

    print(cal_Q(communities, G))
    print(f'算法执行时间{end_time - start_time}')
    # 可视化结果
    showCommunity(G, communities, pos)
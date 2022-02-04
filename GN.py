import time
import networkx as nx
import matplotlib.pyplot as plt


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


# 克隆图数据集
def cloned_graph(G):
    cloned_g = nx.Graph()
    for edge in G.edges():
        cloned_g.add_edge(edge[0], edge[1])
    return cloned_g


# 计算模块度
def cal_Q(partition, G):
    # m代表图中边的数目
    m = len(G.edges(None, False))
    # 利用模块度的化简公式计算
    a = []
    e = []

    # a表示社区内部的点所关联的所有的边的数目与总边数的比例。
    for community in partition:
        t = 0.0
        for node in community:
            t += len(list(G.neighbors(node)))
        a.append(t/(2*m))

    # e表示的是节点全在社区i内部中的边所占的比例
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if(G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t/(2*m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai**2)

    return q


class GN:
    def __init__(self, G):
        self._G = G
        self._G_cloned = cloned_graph(G)
        # 初始划分情况为所有节点为一个社区
        self._partition = [[n for n in G.nodes()]]
        self._max_Q = 0.0

    def execute(self):
        while len(self._G.edges()) != 0:
            # 1.计算每一条边的边介数
            # nx.edge_betweenness返回边介数字典，items返回可遍历的(键, 值) 元组数组。这里每个item是((vi,vj), edge_betweenness))
            # 因此我们取的item[1]最大的item，再取该最小item的[0]，为我们要删除的两个点(即要删除的边)
            edge = max(nx.edge_betweenness_centrality(self._G).items(),
                       key=lambda item: item[1])[0]
            # 2. 移去边介数最大的边
            self._G.remove_edge(edge[0], edge[1])
            # 获得移去边后的子连通图
            components = [list(c)
                          for c in list(nx.connected_components(self._G))]
            if len(components) != len(self._partition):
                # 3. 计算Q值
                cur_Q = cal_Q(components, self._G_cloned)
                if cur_Q > self._max_Q:
                    self._max_Q = cur_Q
                    self._partition = components
        print(self._max_Q)
        print(self._partition)
        return self._partition


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


if __name__ == '__main__':
    # 加载数据集并可视化
    G = load_graph('data/club.txt')
    # print(len(G.nodes(False)))
    # print(len(G.edges(None, False)))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()

    # GN算法
    start_time = time.time()
    algorithm = GN(G)
    partition = algorithm.execute()
    end_time = time.time()
    print(f'算法执行时间{end_time - start_time}')

    # 可视化结果
    showCommunity(algorithm._G_cloned, partition, pos)

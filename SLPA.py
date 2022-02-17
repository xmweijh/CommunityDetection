import collections
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SLPA:
    def __init__(self, G, T, r):
        """
        :param G:图本省
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._r = r

    def execute(self):
        # 将图中数据录入到数据字典中以便使用
        weight = {j: {} for j in self._G.nodes()}
        for q in weight.keys():
            for m in self._G[q].keys():
                # weight[q][m] = self._G[q][m]['weight']
                weight[q][m] = 1
        # 建立成员标签记录  初始本身标签为1
        memory = {i: {i: 1} for i in self._G.nodes()}
        # 开始遍历T次所有节点
        for t in range(self._T):
            listenerslist = list(self._G.nodes())
            # 随机排列遍历顺序
            np.random.shuffle(listenerslist)
            # 开始遍历节点
            for listener in listenerslist:
                # 每个节点的key就是与他相连的节点标签名
                # speakerlist = self._G[listener].keys()
                labels = collections.defaultdict(int)
                # 遍历所有与其相关联的节点
                for speaker in self._G.neighbors(listener):
                    total = float(sum(memory[speaker].values()))
                    # 查看speaker中memory中出现概率最大的标签并记录，key是标签名，value是Listener与speaker之间的权
                    # multinomial从多项式分布中提取样本。
                    # 多项式分布是二项式分布的多元推广。做一个有P个可能结果的实验。这种实验的一个例子是掷骰子，结果可以是1到6。
                    # 从分布图中提取的每个样本代表n个这样的实验。其值x_i = [x_0，x_1，…，x_p] 表示结果为i的次数。
                    # 函数语法
                    # numpy.random.multinomial(n, pvals, size=None)
                    #
                    # 参数
                    # n :  int：实验次数
                    # pvals：浮点数序列，长度p。P个不同结果的概率。这些值应该和为1（但是，只要求和（pvals[：-1]）<=1，最后一个元素总是被假定为考虑剩余的概率）。
                    # size :  int 或 int的元组，可选。 输出形状。如果给定形状为（m，n，k），则绘制 m*n*k 样本。默认值为无，在这种情况下返回单个值。
                    labels[list(memory[speaker].keys())[
                        np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += \
                    weight[listener][speaker]
                # 查看labels中值最大的标签，让其成为当前listener的一个记录
                maxlabel = max(labels, key=labels.get)
                if maxlabel in memory[listener]:
                    memory[listener][maxlabel] += 1
                else:
                    memory[listener][maxlabel] = 1.5
        # 提取出每个节点memory中记录标签出现最多的一个
        # for primary in memory:
        #     p = list(memory[primary].keys())[
        #         np.random.multinomial(1, [freq / total for freq in memory[primary].values()]).argmax()]
        #     memory[primary] = {p: memory[primary][p]}

        for m in memory.values():
            sum_label = sum(m.values())
            threshold_num = sum_label * self._r
            for k, v in list(m.items()):
                if v < threshold_num:
                    del m[k]

        communities = collections.defaultdict(lambda: list())
        # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in memory.items():
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
    start_time = time.time()
    algorithm = SLPA(G, 20, 0.5)
    communities = algorithm.execute()
    end_time = time.time()
    for community in communities:
        print(community)

    print(cal_EQ(communities, G))
    # 可视化结果
    # showCommunity(G, communities, pos)
    print(f'算法执行时间{end_time - start_time}')
import collections
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


def get_percolated_cliques(G, k):
    cliques = list(frozenset(c) for c in nx.find_cliques(G) if len(c) >= k)  # 找出所有大于k的最大k-派系

    #     print(cliques)
    matrix = np.zeros((len(cliques), len(cliques)))  # 构造全0的重叠矩阵
    #     print(matrix)
    for i in range(len(cliques)):
        for j in range(len(cliques)):
            if i == j:  # 将对角线值大于等于k的值设为1，否则设0
                n = len(cliques[i])
                if n >= k:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
            else:  # 将非对角线值大于等于k的值设为1，否则设0
                n = len(cliques[i].intersection(cliques[j]))
                if n >= k - 1:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0

    #     print(matrix)
    #     for i in matrix:
    #         print(i)

    #     l = [-1]*len(cliques)
    l = list(range(len(cliques)))  # l（社区号）用来记录每个派系的连接情况，连接的话值相同
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1 and i != j:  # 矩阵值等于1代表，行派系与列派系相连，让l中的行列派系社区号变一致
                l[j] = l[i]  # 让列派系与行派系社区号相同（划分为一个社区）
    #     print(l)
    q = []  # 用来保存有哪些社区号
    for i in l:
        if i not in q:  # 每个号只取一次
            q.append(i)
    #     print(q)

    p = []  # p用来保存所有社区
    for i in q:
        print(frozenset.union(*[cliques[j] for j in range(len(l)) if l[j] == i]))  # 每个派系的节点取并集获得社区节点
        p.append(list(frozenset.union(*[cliques[j] for j in range(len(l)) if l[j] == i])))
    return p


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


def cal_EQ(cover, G):
    # 存储每个节点所在的社区
    vertex_community = collections.defaultdict(lambda: set())
    # i为社区编号(第几个社区) c为该社区中拥有的节点
    for i, c in enumerate(cover):
        # v为社区中的某一个节点
        for v in c:
            # 根据节点v统计他所在的社区i有哪些
            vertex_community[v].add(i)

    m = 0.0
    for v in G.nodes():
        for n in G.neighbors(v):
            if v > n:
                m += 1

    # m = len(G.edges(None, False))

    total = 0.0
    # 遍历社区
    for c in cover:
        # 遍历社区中的节点i
        for i in c:
            # o_i表示i节点所同时属于的社区数目
            o_i = len(vertex_community[i])
            # k_i表示i节点的度数(所关联的边数)
            k_i = len(G[i])
            # 遍历社区中的节点j
            for j in c:
                # o_j表示j节点所同时属于的社区数目
                o_j = len(vertex_community[j])
                # k_j表示j节点的度数(所关联的边数)
                k_j = len(G[j])
                # 对称情况后面乘以2就行
                if i > j:
                    continue
                t = 0.0
                # 计算公式前半部分  即相邻的点除以重叠度
                if j in G[i]:
                    t += 1.0 / (o_i * o_j)
                # 计算公式后半部分
                t -= k_i * k_j / (2 * m * o_i * o_j)
                if i == j:
                    total += t
                else:
                    total += 2 * t

    return round(total / (2 * m), 4)


def cal_EQ2(cover, G):
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


def add_group(p, G):
    num = 0
    nodegroup = {}
    for partition in p:
        for node in partition:
            nodegroup[node] = {'group': num}
        num = num + 1
    nx.set_node_attributes(G, nodegroup)


def setColor(G):
    color_map = []
    color = ['red', 'green', 'yellow', 'pink', 'blue', 'grey', 'white', 'khaki', 'peachpuff', 'brown']
    for i in G.nodes.data():
        if 'group' not in i[1]:
            color_map.append(color[9])
        else:
            color_map.append(color[i[1]['group']])
    return color_map


# G = load_graph('data/club.txt')
G = load_graph('data/dolphin.txt')
start_time = time.time()
p = get_percolated_cliques(G, 3)
end_time = time.time()
# print(cal_Q(p, G))
print(cal_EQ(p, G))
# add_group(p, G)
# nx.draw(G, with_labels=True, node_color=setColor(G))
# plt.show()
print(f'算法执行时间{end_time - start_time}')
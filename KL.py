import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import kernighan_lin_bisection


def draw_spring(G, com):
    """
    G:图
    com：划分好的社区
    node_size表示节点大小
    node_color表示节点颜色
    node_shape表示节点形状
    with_labels=True表示节点是否带标签
    """
    pos = nx.spring_layout(G)  # 节点的布局为spring型
    NodeId = list(G.nodes())
    node_size = [G.degree(i) ** 1.2 * 90 for i in NodeId]  # 节点大小

    plt.figure(figsize=(8, 6))  # 图片大小
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='w', node_shape='.')

    color_list = ['pink', 'orange', 'r', 'g', 'b', 'y', 'm', 'gray', 'black', 'c', 'brown']
    # node_shape = ['s','o','H','D']

    for i in range(len(com)):
        nx.draw_networkx_nodes(G, pos, nodelist=com[i], node_color=color_list[i])
    plt.show()


if __name__ == "__main__":
    G = nx.karate_club_graph()  # 空手道俱乐部
    # KL算法
    com = list(kernighan_lin_bisection(G))
    print('社区数量', len(com))
    print(com)
    draw_spring(G, com)

import numpy as np
import networkx as nx
from heapq import heappush, heappop
from matplotlib import pyplot as plt
import copy
import time


def walktrap(G, t, verbose=False):
    class Community:
        def __init__(self, new_C_id, C1=None, C2=None):
            self.id = new_C_id
            # 一个节点作为一个社区
            if C1 is None:
                self.size = 1
                self.P_c = P_t[self.id]  # probab vector
                self.adj_coms = {}
                self.vertices = set([self.id])
                self.internal_weight = 0.
                self.total_weight = self.internal_weight + (len([id for id, x in enumerate(A[self.id]) if
                                                                 x == 1. and id != self.id]) / 2.)  # External edges have 0.5 weight, ignore edge to itself
            # 合并形成新社区
            else:
                self.size = C1.size + C2.size
                self.P_c = (C1.size * C1.P_c + C2.size * C2.P_c) / self.size
                # Merge info about adjacent communities, but remove C1, C2
                self.adj_coms = dict(C1.adj_coms.items() | C2.adj_coms.items())
                del self.adj_coms[C1.id]
                del self.adj_coms[C2.id]
                self.vertices = C1.vertices.union(C2.vertices)
                weight_between_C1C2 = 0.
                for v1 in C1.vertices:
                    for id, x in enumerate(A[v1]):
                        if x == 1. and id in C2.vertices:
                            weight_between_C1C2 += 1.
                self.internal_weight = C1.internal_weight + C2.internal_weight + weight_between_C1C2
                self.total_weight = C1.total_weight + C2.total_weight

        def modularity(self):
            # 模块度计算
            return (self.internal_weight - (self.total_weight * self.total_weight / G_total_weight)) / G_total_weight

    # 获得节点数目
    N = G.number_of_nodes()
    # 获得邻接矩阵
    A = np.array(nx.to_numpy_matrix(G))

    # 转移矩阵P 以及 对角度矩阵D
    Dx = np.zeros((N, N))
    P = np.zeros((N, N))
    # 遍历邻接矩阵的每一行 对应每个节点与其他节点的邻接关系
    for i, A_row in enumerate(A):
        # 邻接矩阵一行的和为对应节点的度
        d_i = np.sum(A_row)
        # 转移概率 等于每个邻居节点的权重除以该节点的度
        P[i] = A_row / d_i
        # 后面计算都是D-0.5 提前处理好方便后面计算
        Dx[i, i] = d_i ** (-0.5)

    # 采用t次随机游走
    P_t = np.linalg.matrix_power(P, t)

    # 边的总权重
    G_total_weight = G.number_of_edges()

    # 当前的社区
    community_count = N
    communities = {}
    # 刚开始将每个节点作为一个社区
    for C_id in range(N):
        communities[C_id] = Community(C_id)

    # 储存Δσ: <deltaSigma(C1,C2), C1_id, C2_id>
    min_sigma_heap = []
    # 遍历所有相邻节点 计算Δσ
    for e in G.edges:
        C1_id = e[0]
        C2_id = e[1]
        if C1_id != C2_id:
            # 利用公式计算 这里|c1||c2|\(|c1|+|c2|)个数都是1直接等于0.5
            ds = (0.5 / N) * np.sum(np.square(np.matmul(Dx, P_t[C1_id]) - np.matmul(Dx, P_t[C2_id])))
            # 利用堆排序存储
            heappush(min_sigma_heap, (ds, C1_id, C2_id))
            # 更新每个社区以及它的邻居社区
            communities[C1_id].adj_coms[C2_id] = ds
            communities[C2_id].adj_coms[C1_id] = ds

    delta_sigmas = []

    partitions = [set(np.arange(N))]
    # 第一次划分时每个节点为自身的社区
    # 计算初始模块度Q
    modularities = [np.sum([communities[C_id].modularity() for C_id in partitions[0]])]
    if verbose:
        print("Partition 0: ", partitions[0])
        print("Q(0) = ", modularities[0])

    # 开始迭代
    for k in range(1, N):
        # 根据最小的Δσ合并C1,C2
        # 需要确保最小的Δσ的C1,C2仍然在当前的划分社区列表中
        while min_sigma_heap:
            # 取出当前最小的Δσ
            delta_sigma_C1C2, C1_id, C2_id = heappop(min_sigma_heap)
            if C1_id in partitions[k - 1] and C2_id in partitions[k - 1]:
                break
        # Record delta sigma at this step
        delta_sigmas.append(delta_sigma_C1C2)

        # 合并C1,C2为C3, 分配id
        C3_id = community_count
        community_count += 1  # increase for the next one
        communities[C3_id] = Community(C3_id, communities[C1_id], communities[C2_id])

        # 添加新的划分(k-th)
        partitions.append(copy.deepcopy(partitions[k - 1]))
        partitions[k].add(C3_id)  # add C3_ID
        partitions[k].remove(C1_id)
        partitions[k].remove(C2_id)

        # 更新C3和以前C1或C2的邻居社区之间的delta_sigma_heap
        # 遍历C3的邻居社区
        for C_id in communities[C3_id].adj_coms.keys():
            # 如果C是C1,C2共同的邻居则用公式4
            if (C_id in communities[C1_id].adj_coms) and (C_id in communities[C2_id].adj_coms):
                delta_sigma_C1C = communities[C1_id].adj_coms[C_id]
                delta_sigma_C2C = communities[C2_id].adj_coms[C_id]
                # 使用公式4 to (C, C3)
                ds = ((communities[C1_id].size + communities[C_id].size) * delta_sigma_C1C + (
                        communities[C2_id].size + communities[C_id].size) * delta_sigma_C2C - communities[
                          C_id].size * delta_sigma_C1C2) / (communities[C3_id].size + communities[C_id].size)

            # 否则使用公式3 to (C, C3)
            else:
                ds = np.sum(np.square(np.matmul(Dx, communities[C_id].P_c) - np.matmul(Dx, communities[C3_id].P_c))) * \
                     communities[C_id].size * communities[C3_id].size / (
                             (communities[C_id].size + communities[C3_id].size) * N)

            # 更新min_sigma_heap以及更新C3,C之间的delta sigmas
            heappush(min_sigma_heap, (ds, C3_id, C_id))
            communities[C3_id].adj_coms[C_id] = ds
            communities[C_id].adj_coms[C3_id] = ds

            # 计算并存储当前划分的模块度
        modularities.append(np.sum([communities[C_id].modularity() for C_id in partitions[k]]))

        if verbose:
            print("Partition ", k, ": ", partitions[k])
            print("\tMerging ", C1_id, " + ", C2_id, " --> ", C3_id)
            print("\tQ(", k, ") = ", modularities[k])
            print("\tdelta_sigma = ", delta_sigmas[k - 1])

    return np.array(partitions), communities, np.array(delta_sigmas), np.array(modularities)


# 计算 Rand index
def calculate_rand_index(P1, P2):
    N = 0
    sum_intersect = 0.
    sum_C1 = 0.
    sum_C2 = np.sum([len(s) ** 2 for s in P2])
    for s1 in P1:
        N += len(s1)
        sum_C1 += len(s1) ** 2
        for s2 in P2:
            sum_intersect += len(s1.intersection(s2)) ** 2
    return (N * N * sum_intersect - sum_C1 * sum_C2) / (0.5 * N * N * (sum_C1 + sum_C2) - sum_C1 * sum_C2)


def partition_to_plot(coms, partition):
    p_dict = {}
    for i, C_id in enumerate(partition):
        for v in coms[C_id].vertices:
            p_dict[v] = i
    return p_dict


def partition_dict_to_sets(d):
    inverse_dict = {}
    for k, v in d.items():
        if v in inverse_dict:
            inverse_dict[v].add(k)
        else:
            inverse_dict[v] = set([k])

    return inverse_dict.values()


def partition_set_to_sets(comms, partition):
    list_of_sets = []
    for C_id in partition:
        list_of_sets.append(copy.deepcopy(comms[C_id].vertices))
    return list_of_sets


if __name__ == '__main__':
    G = nx.read_edgelist("data/football.txt")
    k = 12
    G = nx.convert_node_labels_to_integers(G)
    pos = nx.spring_layout(G)

    t = 2
    parts, coms, _, Qs = walktrap(G, t)
    Qmax_index = np.argmax(Qs)
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    sort_p = sorted(my_best_part.items(), key=lambda x: x[0])
    # print([x[1] for x in sort_p])
    nx.draw(G, pos, node_color=[x[1] for x in sort_p])
    plt.show()
    print(my_best_part)

    print(partition_dict_to_sets(my_best_part))
    print(calculate_rand_index(partition_dict_to_sets(my_best_part), partition_dict_to_sets(my_best_part)))
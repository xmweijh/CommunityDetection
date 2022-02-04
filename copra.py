# coding=utf-8
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random
import numpy as np


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

    def Propagate(self, x, old, new, v, asynchronous):
        # 依据邻结点标签集更新该节点
        for eachpoint in self._G.neighbors(x):
            for eachlable in old[eachpoint]:
                b = old[eachpoint][eachlable]
                if eachlable in new[x]:
                    new[x][eachlable] += b
                else:
                    new[x].update({eachlable: b})
                if asynchronous:
                    old[x] = copy.deepcopy(new[x])
        Normalize(new[x])
        # 存储最大b值
        maxb = 0.0
        # 存储最大b值的节点序号
        maxc = 0
        # 记录需要删除的节点序号
        t = []
        # 去除小于1/v的候选项，若均小于则''选b最大的赋值''  若都小于取最大的一个
        for each in new[x]:
            if new[x][each] < 1 / float(v):
                t.append(each)
                if new[x][each] >= maxb:  # 取最后一个
                    maxb = new[x][each]
                    maxc = each
        for i in range(len(t)):
            del new[x][t[i]]
        if len(new[x]) == 0:
            new[x][maxc] = 1
        else:
            self.Normalize(new[x])

    def Normalize(self, x):
        sums = 0.0
        for each in x:
            sums += x[each]
        for each in x:
            if sums != 0:
                x[each] = x[each] / sums

    def id_l(self, l):
        ids = []
        for each in l:
            ids.append(id_x(each))
        return ids

    def id_x(self, x):
        ids = []
        for each in x:
            ids.append(each)
        return ids

    def count(self, l):
        counts = {}
        for eachpoint in l:
            for eachlable in eachpoint:
                if eachlable in counts:
                    n = counts[eachlable]
                    counts.update({eachlable: n + 1})
                else:
                    counts.update({eachlable: 1})
        return counts

    def mc(self, cs1, cs2):
        # print cs1,cs2
        cs = {}
        for each in cs1:
            if each in cs2:
                cs[each] = min(cs1[each], cs2[each])
        return cs

    def execute(self):
        label_new = [{} for i in self._G.nodes()]
        label_old = [{i: 1} for i in self._G.nodes()]
        minl = {}
        oldmin = {}
        flag = True  # asynchronous
        itera = 0  # 迭代次数
        start = time.perf_counter()  # 计时

        visitlist = list(self._G.nodes())
        # 随机排列遍历顺序
        np.random.shuffle(visitlist)
        # 同异步迭代过程
        while True:
            '''
            if flag:
                flag = False
            else:
                flag = True
            '''
            itera += 1
            for each in visitlist:
                self.Propagate(each, label_old, label_new, self._v, flag)
            if self.id_l(label_old) == self.id_l(label_new):
                inl = self.mc(minl, label_new)
            else:
                minl = label_new
            if minl != oldmin:
                label_old = label_new
                oldmin = minl
            else:
                break
        print(itera, label_old)
        coms = {}
        sub = {}
        for each in range(vertices):
            ids = id_x(label_old[each])
            for eachc in ids:
                if eachc in coms and eachc in sub:
                    coms[eachc].append(each)
                    # elif :
                    sub.update({eachc: set(sub[eachc]) & set(ids)})
                else:
                    coms.update({eachc: [each]})
                    sub.update({eachc: ids})
        print('lastiter', coms)
        # 获取每个节点属于的标签数
        o = [0 for i in range(vertices)]
        for eachid in range(vertices):
            for eachl in coms:
                if eachid in coms[eachl]:
                    o[eachid] += 1
                    # 去重取标签
        for each in sub:
            if len(sub[each]):
                for eachc in sub[each]:
                    if eachc != each:
                        coms[eachc] = list(set(coms[eachc]) - set(coms[each]))
        # 标签整理
        clusterment = [0 for i in range(vertices)]
        a = 0
        for eachc in coms:
            if len(coms[eachc]) != 0:
                for e in coms[eachc]:
                    clusterment[e] = a + 1
                a += 1
        degree_s = sorted(degree_s, key=lambda x: x[0], reverse=False)
        elapsed = (time.perf_counter() - start)
        print('t=', elapsed)
        print('result=', coms)
        print('clusterment=', clusterment)
        print('Q =', Modulartiy(A, coms, sums, vertices))
        print('EQ =', ExtendQ(A, coms, sums, degree_s, o))
        # print 'NMI=',NMI(coms,coms)
        return coms


if __name__ == '__main__':
    # 节点个数,V
    vertices = [34, 115, 1589, 62]
    # txtlist = ['karate.txt','football.txt','science.txt','dolphins.txt']
    txtlist = ['karate.txt']
    # vertices = [64,128,256,512]
    # txtlist = ['RN1.txt','RN2.txt','RN3.txt','RN4.txt']
    testv = [2, 3, 4, 5]
    for i in range(len(txtlist)):
        print("txt name: <<{}>> vertices num: {}".format(txtlist[i], vertices[i]))
        for ev in testv:
            print('v =', ev)
            A = LoadAdjacentMatrixData(txtlist[i], vertices[i])
            degree_s, neighbours, sums = Degree_Sorting(A, vertices[i])
            # print neighbours
            getcoms(degree_s, neighbours, sums, A, ev, vertices[i])

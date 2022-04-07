import igraph as ig
import leidenalg
import louvain

# 按照边列表的形式读入文件，生成无向图
g = ig.Graph.Read_Edgelist("data//OpenFlights.txt", directed=False)


part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
print(part)
print(ig.summary(g))
print(part.modularity)
ig.plot(part)

part2 = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, resolution_parameter=0.01)
print(part2.modularity)
# ig.plot(part2)


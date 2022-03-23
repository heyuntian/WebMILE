import logging
from defs import MILEAPIControl
from graph import Graph
from utils import cmap2C, updateCtrl, parse_args
from collections import defaultdict
import numpy as np

def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt

def jaccard_idx_preprocess(ctrl, graph, matched, groups):
    '''Use hashmap to find out nodes with exactly same neighbors.'''
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        neighs2node[neighs].append(i)
    for key in sorted(neighs2node.keys()):
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def generate_hybrid_matching(ctrl, graph):
    """
    Generate matchings using the hybrid method. It changes the cmap in graph object,
    return groups array and coarse_graph_size.
    """
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)

    max_node_wgt = ctrl.max_node_wgt

    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(ctrl, graph, matched, groups)
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree, kind='mergesort')
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node.
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return groups, coarse_graph_size


def create_coarse_graph(ctrl, graph, groups, coarse_graph_size):
    '''create the coarser graph and return it based on the groups array and coarse_graph_size'''
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree

    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    for idx in range(len(groups)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop.
                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges

    coarse_graph.edge_num = nedges

    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    return coarse_graph


def print_coarsen_info(ctrl, g):
    """
    Display the information of the series of coarsened graphs
    """
    cnt = 0
    while g is not None:
        ctrl.logger.info("Level " + str(cnt) + " --- # nodes: " + str(g.node_num))
        g = g.coarser
        cnt += 1


def coarsen(ctrl, graph):
    """
    Input: ctrl, graph
    Output: coarsened_graph
    """
    # Step-1: Graph Coarsening.
    original_graph = graph
    coarsen_level = ctrl.coarsen_level
    # if ctrl.refine_model.double_base:  # if it is double-base, it will need to do one more layer of coarsening
    #     coarsen_level += 1
    for i in range(coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(ctrl, graph)
        # print(graph.cmap[:10], match[:10])
        coarse_graph = create_coarse_graph(ctrl, graph, match, coarse_graph_size)
        graph = coarse_graph
        # print(graph.node_num, graph.edge_num, graph.adj_idx[:10])
        if graph.node_num <= ctrl.embed_dim:
            ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
            exit(0)

    # if ctrl.debug_mode and graph.node_num < 1e3:
    #     assert np.allclose(graph_to_adj(graph).A, graph.A.A), "Coarser graph is not consistent with Adj matrix"
    print_coarsen_info(ctrl, original_graph)
    return graph

def output_coarsened(ctrl, graph):
    """
    Output the coarsened graph in the format of edgelist.
    :param ctrl:
    :param graph:
    :return: Path of output coarsened graph.
    """
    n, m, adj_list, adj_idx, adj_wgt = graph.node_num, graph.edge_num, graph.adj_list, graph.adj_idx, graph.adj_wgt
    if ctrl.output_format == 'edgelist':
        graph_file = open(ctrl.coarsen_path, "w")
        for u in range(n):
            for j in range(adj_idx[u], adj_idx[u + 1]):
                v, w = adj_list[j], adj_wgt[j]
                graph_file.write(f"{u} {v} {w}\n" if u <= v else f"{v} {u} {w}\n")
        graph_file.close()


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)

    ctrl = MILEAPIControl()
    args = parse_args(useCoarsen=True)
    graph, mapping = updateCtrl(ctrl, args, useCoarsen=True)

    coarsened = coarsen(ctrl, graph)
    output_coarsened(ctrl, coarsened)


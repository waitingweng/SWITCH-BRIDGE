from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from igraph import Graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools


def module_relation_bipartite(q_value, first_module, second_module, first_module_stage, second_module_stage, save=False):
    # Example mapping between vertex names and sizes
    module_sizes_mapping = dict()
    overlap_pairs = list()
    p_value_list = list()

    # Size of the population (total number of genes)
    total_genes = 12939

    for i in first_module:
        for j in second_module:
            # Size of the first gene module
            module_1_size = len(first_module[i])
            module_sizes_mapping[i] = module_1_size
            # Size of the second gene module
            module_2_size = len(second_module[j])
            module_sizes_mapping[j] = module_2_size

            # Number of overlapping genes between the two modules
            overlap_size = len(set(first_module[i]) & set(second_module[j]))
            if overlap_size <= 3:
                continue
            union_size = len(set(first_module[i]) | set(second_module[j]))

            # Perform hypergeometric test
            p_value = hypergeom.sf(
                overlap_size-1, total_genes, module_1_size, module_2_size)
            p_value_list.append(p_value)
            overlap_pairs.append((i, j, overlap_size / union_size))

    # Perform multiple testing adjustment
    _, q_values, _, _ = multipletests(p_value_list, method='fdr_bh')
    relation_pairs = list()
    for i in range(len(q_values)):
        if q_values[i] < q_value:
            relation_pairs.append(overlap_pairs[i])

    # Create a weighted graph directly from the edge list
    graph = Graph.TupleList(relation_pairs, directed=False, weights=True)
    # Assign vertex sizes using the mapping
    vertex_sizes = [module_sizes_mapping.get(
        name) for name in graph.vs["name"]]
    # Assign vertex sizes to the graph
    graph.vs["size"] = vertex_sizes
    # Print the graph summary
    # print(graph.summary())
    if save == True:
        graph.write_graphml(
            f"module_{first_module_stage}_vs_{second_module_stage}_relation.graphml")
    return graph


def size_MDI(bipartite_relation_graph, first_stage="", second_stage=""):
    size_index = 0
    # max_size = max(bipartite_relation_graph.vs["size"])
    # min_size = min(bipartite_relation_graph.vs["size"])

    for edge in bipartite_relation_graph.es:
        source_name = bipartite_relation_graph.vs[edge.source]["name"]
        target_name = bipartite_relation_graph.vs[edge.target]["name"]

        if source_name.split("_")[1] == second_stage:
            source_name, target_name = target_name, source_name

        # source_norm_size = (bipartite_relation_graph.vs.find(name=source_name)[
        #                     "size"] - min_size) / (max_size - min_size)
        # target_norm_size = (bipartite_relation_graph.vs.find(name=target_name)[
        #                     "size"] - min_size) / (max_size - min_size)

        # print(source_norm_size, target_norm_size, source_norm_degree, target_norm_degree)
        weight = edge["weight"]
        # size_index += weight * (target_norm_size-source_norm_size)
        size_index += weight * (bipartite_relation_graph.vs.find(name=source_name)[
                                "size"] - bipartite_relation_graph.vs.find(name=target_name)["size"])

    # print(size_index)
    return size_index


def degree_MDI(bipartite_relation_graph, first_stage="", second_stage=""):
    degree_index = 0
    # max_degree = max(bipartite_relation_graph.degree())
    # min_degree = min(bipartite_relation_graph.degree())

    for edge in bipartite_relation_graph.es:
        source_name = bipartite_relation_graph.vs[edge.source]["name"]
        target_name = bipartite_relation_graph.vs[edge.target]["name"]

        if source_name.split("_")[1] == second_stage:
            source_name, target_name = target_name, source_name

        # source_norm_degree = (bipartite_relation_graph.degree(bipartite_relation_graph.vs.find(
        #     name=source_name)) - min_degree) / (max_degree - min_degree)
        # target_norm_degree = (bipartite_relation_graph.degree(bipartite_relation_graph.vs.find(
        #     name=target_name)) - min_degree) / (max_degree - min_degree)

        # print(source_norm_size, target_norm_size, source_norm_degree, target_norm_degree)
        weight = edge["weight"]
        # degree_index += weight * (target_norm_degree-source_norm_degree)
        degree_index += weight * (bipartite_relation_graph.degree(bipartite_relation_graph.vs.find(
            name=source_name)) - bipartite_relation_graph.degree(bipartite_relation_graph.vs.find(name=target_name)))

    # print(degree_index)
    return degree_index


# measure the proportion of genes that are preserved during the transition
def conservation_MDI(bipartite_relation_graph, first_module, second_module, first_stage="", second_stage=""):
    conservation_pairs = list()
    max_conservation_ratio = 0
    min_conservation_ratio = 1

    conservation_index = 0

    for edge in bipartite_relation_graph.es:
        source_name = bipartite_relation_graph.vs[edge.source]["name"]
        target_name = bipartite_relation_graph.vs[edge.target]["name"]

        if source_name.split("_")[1] == second_stage:
            source_name, target_name = target_name, source_name

        source_neighbors = bipartite_relation_graph.neighbors(
            bipartite_relation_graph.vs.find(name=source_name))
        source_neighbor_names = [
            bipartite_relation_graph.vs[neighbor]["name"] for neighbor in source_neighbors]
        source_size = len(first_module[source_name])
        source = set(first_module[source_name])
        source_conservation = set()
        for i in source_neighbor_names:
            source_conservation |= set(second_module[i])
        source_conservation &= source
        # print(source_conservation)
        source_conservation_ratio = len(source_conservation)/source_size

        target_neighbors = bipartite_relation_graph.neighbors(
            bipartite_relation_graph.vs.find(name=target_name))
        target_neighbor_names = [
            bipartite_relation_graph.vs[neighbor]["name"] for neighbor in target_neighbors]
        target_size = len(second_module[target_name])
        target = set(second_module[target_name])
        target_conservation = set()
        for i in target_neighbor_names:
            target_conservation |= set(first_module[i])
        target_conservation &= target
        # print(target_conservation)
        target_conservation_ratio = len(target_conservation)/target_size

        weight = edge["weight"]
        conservation_pairs.append(
            [weight, source_conservation_ratio, target_conservation_ratio])
        # max_conservation_ratio = max(
        #     max_conservation_ratio, source_conservation_ratio, target_conservation_ratio)
        # min_conservation_ratio = min(
        #     min_conservation_ratio, source_conservation_ratio, target_conservation_ratio)

    for weight, source_conservation_ratio, target_conservation_ratio in conservation_pairs:
        conservation_index += weight * \
            (source_conservation_ratio - target_conservation_ratio)

    # print(conservation_index)
    return conservation_index


def construct_parent(parent_name, stage_module, pcc_matrix, pcc_theshold=0.7):

    gene_symbols = stage_module[parent_name]

    # Threshold for including edges
    threshold = pcc_theshold

    stage_module_pcc = pcc_matrix.loc[gene_symbols, gene_symbols]

    # Convert PCC matrix to binary adjacency matrix
    adjacency_matrix = (abs(stage_module_pcc) > threshold).astype(int)

    # Create an unweighted graph
    graph = Graph.Weighted_Adjacency(
        adjacency_matrix, mode="UNDIRECTED", attr="weight", loops=False)

    # Set vertex names
    graph.vs["name"] = gene_symbols

    components = graph.connected_components()
    connected_graph = components.giant()

    return connected_graph


def construct_children(parent_name, parent_graph, relation_graph, stage_module):
    parent_graph = parent_graph.copy()
    children_name = list()
    children_id = relation_graph.neighbors(parent_name)
    for id in children_id:
        children_name.append(relation_graph.vs.find(id)["name"])

    children_overlapped_parts = dict()
    for i in children_name:
        children_overlapped_parts[i] = set(
            parent_graph.vs["name"]) & set(stage_module[i])

    return children_overlapped_parts


def delete_children_connecting_edges(parent_graph, children_overlapped_parts):
    parent_graph = parent_graph.copy()

    from itertools import combinations
    children_name = list(children_overlapped_parts.keys())

    deleted_edge_list = list()

    # Get all combinations of items taken 2 at a time
    all_combinations = list(combinations(children_name, 2))
    for i, j in all_combinations:
        # print((i, j))
        for gene1 in children_overlapped_parts[i]:
            for gene2 in children_overlapped_parts[j]:
                # print(gene1, gene2)
                if parent_graph.are_connected(gene1, gene2):
                    deleted_edge_list.append((gene1, gene2))

    return deleted_edge_list


def deleted_edges_to_nodes(deleted_edge_list):
    from copy import deepcopy
    edge_list = deepcopy(deleted_edge_list)
    sum_edge = 0
    nodes = list()
    while sum_edge < len(deleted_edge_list):
        degree_dict = dict()
        for i in edge_list:
            degree_dict.setdefault(i[0], 0)
            degree_dict.setdefault(i[1], 0)
            degree_dict[i[0]] += 1
            degree_dict[i[1]] += 1

        max_degree = max(degree_dict.items(), key=lambda x: x[1])
        sum_edge += max_degree[1]
        nodes.append(max_degree[0])
        edge_list = [i for i in edge_list if i[0] !=
                     max_degree[0] and i[1] != max_degree[0]]

    return nodes


def merge_parent(parent_graph, children_overlapped_parts):
    merged_parent_graph = parent_graph.copy()
    for name, genes_to_merge in children_overlapped_parts.items():
        merged_parent_graph.add_vertex(name=name)
        name_id = merged_parent_graph.vs.find(name=name).index
        for gene in genes_to_merge:

            # Get outgoing edges from vertex 0
            neighbors = merged_parent_graph.neighbors(gene)
            # print(len(neighbors))
            # print(neighbors)
            edge_list = list()
            weight_dict = {"weight": []}
            # Iterate through outgoing edges and add equivalent edges to the new vertex
            for n in neighbors:
                source = name_id  # Change the source to the new vertex
                target = n
                weight = 1
                edge_list.append((source, target))
                weight_dict["weight"].append(weight)
                # print(target)
                # Add the equivalent edge to the new vertex
            merged_parent_graph.add_edges(edge_list, attributes=weight_dict)

        merged_parent_graph.delete_vertices(genes_to_merge)

    merged_parent_graph.simplify(combine_edges=sum)

    return merged_parent_graph


def delete_merged_nodes_connection(merged_parent_graph, children_overlapped_parts):
    from itertools import combinations
    merged_parent_graph = merged_parent_graph.copy()
    children_names = list(children_overlapped_parts.keys())
    deleted_edges = list()
    for i, j in combinations(children_names, 2):
        if merged_parent_graph.are_connected(i, j):
            deleted_edges.append((i, j))

    merged_parent_graph.delete_edges(deleted_edges)
    return merged_parent_graph


def spectral_analysis(deleted_merged_nodes_connections_parent_graph, n_clusters):
    from sklearn.cluster import SpectralClustering
    import numpy as np
    dg = deleted_merged_nodes_connections_parent_graph.copy()
    # Obtain the adjacency matrix and degree matrix with weights
    adjacency_matrix = np.array(dg.get_adjacency(attribute='weight').data)

    # Create an instance of SpectralClustering with precomputed Laplacian
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=123456)

    # Fit and predict clusters
    labels = spectral.fit_predict(adjacency_matrix)

    return labels


def find_splitting_transitional_set(parent_name, first_stage_module, second_stage_module, relation_graph, pcc_matrix, pcc_theshold):
    parent_graph = construct_parent(
        parent_name=parent_name, stage_module=first_stage_module, pcc_matrix=pcc_matrix, pcc_theshold=pcc_theshold)
    # print("parent_graph", parent_graph.summary())

    children_overlapped_parts = construct_children(
        parent_name=parent_name, parent_graph=parent_graph, relation_graph=relation_graph, stage_module=second_stage_module)
    # print("length of children_overlapped_parts", len(children_overlapped_parts))
    N_children = len(children_overlapped_parts)

    deleted_edge_list = delete_children_connecting_edges(
        parent_graph=parent_graph, children_overlapped_parts=children_overlapped_parts)
    # print("length of delted_edge_list", len(deleted_edge_list))

    nodes = deleted_edges_to_nodes(deleted_edge_list=deleted_edge_list)
    # print("length of deleted nodes", len(nodes))

    mg = merge_parent(parent_graph=parent_graph,
                      children_overlapped_parts=children_overlapped_parts)
    # print("merged_parent_graph", mg.summary())

    dg = delete_merged_nodes_connection(
        merged_parent_graph=mg, children_overlapped_parts=children_overlapped_parts)
    # print("deleted_merged_nodes_connections_parent_graph", dg.summary())

    labels = spectral_analysis(
        deleted_merged_nodes_connections_parent_graph=dg, n_clusters=N_children)
    # print(labels[-N_children:])
    # print("split ratio =", len(set(labels[-N_children:])) / N_children)

    return nodes


def find_all_splitting_transitional_set(first_stage_module, second_stage_module, relation_graph, pcc_matrix, pcc_theshold):
    all_nodes = list()
    for parent_name in first_stage_module:
        if parent_name in relation_graph.vs["name"] and relation_graph.degree(parent_name) > 1:
            nodes = find_splitting_transitional_set(parent_name=parent_name, first_stage_module=first_stage_module,
                                                    second_stage_module=second_stage_module, relation_graph=relation_graph, pcc_matrix=pcc_matrix, pcc_theshold=pcc_theshold)
            all_nodes += nodes

    return all_nodes


def find_all_transitional_set(first_stage_module, second_stage_module, relation_graph, first_pcc_matrix, second_pcc_matrix, pcc_theshold):
    all_nodes_split = find_all_splitting_transitional_set(
        first_stage_module=first_stage_module, second_stage_module=second_stage_module, relation_graph=relation_graph, pcc_matrix=first_pcc_matrix, pcc_theshold=pcc_theshold)
    all_nodes_merge = find_all_splitting_transitional_set(
        first_stage_module=second_stage_module, second_stage_module=first_stage_module, relation_graph=relation_graph, pcc_matrix=second_pcc_matrix, pcc_theshold=pcc_theshold)
    mutual_nodes = set(all_nodes_split) & set(all_nodes_merge)

    with open(f'transitional_set_{first_pcc_matrix.name.split("_")[-1]}_vs_{second_pcc_matrix.name.split("_")[-1]}.txt', 'w') as file:
        for item in mutual_nodes:
            file.write(f'{item}\n')
    return mutual_nodes


def module_to_degree_MDI(q_value, module_4S, module_I, module_III, module_IV):
    graph_4S_vs_I = module_relation_bipartite(
        q_value=q_value, first_module=module_4S, second_module=module_I, first_module_stage="4S", second_module_stage="I")
    graph_4S_vs_III = module_relation_bipartite(
        q_value=q_value, first_module=module_4S, second_module=module_III, first_module_stage="4S", second_module_stage="III")
    graph_4S_vs_IV = module_relation_bipartite(
        q_value=q_value, first_module=module_4S, second_module=module_IV, first_module_stage="4S", second_module_stage="IV")
    # graph_4S_vs_NA = module_relation_bipartite(q_value=q_value, first_module=module_4S, second_module=module_NA, first_module_stage="4S", second_module_stage="NA")
    graph_I_vs_III = module_relation_bipartite(
        q_value=q_value, first_module=module_I, second_module=module_III, first_module_stage="I", second_module_stage="III")
    graph_I_vs_IV = module_relation_bipartite(
        q_value=q_value, first_module=module_I, second_module=module_IV, first_module_stage="I", second_module_stage="IV")
    # graph_I_vs_NA = module_relation_bipartite(q_value=q_value, first_module=module_I, second_module=module_NA, first_module_stage="I", second_module_stage="NA")
    graph_III_vs_IV = module_relation_bipartite(
        q_value=q_value, first_module=module_III, second_module=module_IV, first_module_stage="III", second_module_stage="IV")
    # graph_III_vs_NA = module_relation_bipartite(q_value=q_value, first_module=module_III, second_module=module_NA, first_module_stage="III", second_module_stage="NA")
    # graph_IV_vs_NA = module_relation_bipartite(q_value=q_value, first_module=module_IV, second_module=module_NA, first_module_stage="IV", second_module_stage="NA")

    edge_4S_vs_I = degree_MDI(
        bipartite_relation_graph=graph_4S_vs_I, first_stage="4S", second_stage="I")
    edge_4S_vs_III = degree_MDI(
        bipartite_relation_graph=graph_4S_vs_III, first_stage="4S", second_stage="III")
    edge_4S_vs_IV = degree_MDI(
        bipartite_relation_graph=graph_4S_vs_IV, first_stage="4S", second_stage="IV")
    # edge_4S_vs_NA = degree_MDI(bipartite_relation_graph=graph_4S_vs_NA, first_stage="4S", second_stage="NA")
    edge_I_vs_III = degree_MDI(
        bipartite_relation_graph=graph_I_vs_III, first_stage="I", second_stage="III")
    edge_I_vs_IV = degree_MDI(
        bipartite_relation_graph=graph_I_vs_IV, first_stage="I", second_stage="IV")
    # edge_I_vs_NA = degree_MDI(bipartite_relation_graph=graph_I_vs_NA, first_stage="I", second_stage="NA")
    edge_III_vs_IV = degree_MDI(
        bipartite_relation_graph=graph_III_vs_IV, first_stage="III", second_stage="IV")
    # edge_III_vs_NA = degree_MDI(bipartite_relation_graph=graph_III_vs_NA, first_stage="III", second_stage="NA")
    # edge_IV_vs_NA = degree_MDI(bipartite_relation_graph=graph_IV_vs_NA, first_stage="IV", second_stage="NA")

    degree_MDI_array = np.array(
        [edge_4S_vs_I, edge_4S_vs_III, edge_4S_vs_IV, edge_I_vs_III, edge_I_vs_IV, edge_III_vs_IV])

    # print(direction_array)
    return degree_MDI_array


def module_to_error(path, module_4S, module_I, module_III, module_IV):
    degree_MDI_05 = module_to_degree_MDI(
        q_value=0.05, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_01 = module_to_degree_MDI(
        q_value=0.01, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_005 = module_to_degree_MDI(
        q_value=0.005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_001 = module_to_degree_MDI(
        q_value=0.001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0005 = module_to_degree_MDI(
        q_value=0.0005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0001 = module_to_degree_MDI(
        q_value=0.0001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_00005 = module_to_degree_MDI(
        q_value=0.00005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_00001 = module_to_degree_MDI(
        q_value=0.00001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_000005 = module_to_degree_MDI(
        q_value=0.000005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_000001 = module_to_degree_MDI(
        q_value=0.000001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0000005 = module_to_degree_MDI(
        q_value=0.0000005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0000001 = module_to_degree_MDI(
        q_value=0.0000001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)

    # the differences of indices comparing with the first index
    error_05 = sum(np.abs(np.subtract(degree_MDI_05, degree_MDI_05)))
    error_01 = sum(np.abs(np.subtract(degree_MDI_01, degree_MDI_05)))
    error_005 = sum(np.abs(np.subtract(degree_MDI_005, degree_MDI_05)))
    error_001 = sum(np.abs(np.subtract(degree_MDI_001, degree_MDI_05)))
    error_0005 = sum(np.abs(np.subtract(degree_MDI_0005, degree_MDI_05)))
    error_0001 = sum(np.abs(np.subtract(degree_MDI_0001, degree_MDI_05)))
    error_00005 = sum(np.abs(np.subtract(degree_MDI_00005, degree_MDI_05)))
    error_00001 = sum(np.abs(np.subtract(degree_MDI_00001, degree_MDI_05)))
    error_000005 = sum(np.abs(np.subtract(degree_MDI_000005, degree_MDI_05)))
    error_000001 = sum(np.abs(np.subtract(degree_MDI_000001, degree_MDI_05)))
    error_0000005 = sum(np.abs(np.subtract(degree_MDI_0000005, degree_MDI_05)))
    error_0000001 = sum(np.abs(np.subtract(degree_MDI_0000001, degree_MDI_05)))

    errors = [error_05, error_01, error_005, error_001,
              error_0005, error_0001, error_00005,
              error_00001, error_000005, error_000001,
              error_0000005, error_0000001]

    def errors_to_txt(path, errors):
        with open(path, 'w') as file:
            for i in errors:
                file.write(str(i) + "\n")

    errors_to_txt(path=path, errors=errors)

    return errors


def module_to_variation(path, module_4S, module_I, module_III, module_IV):
    degree_MDI_05 = module_to_degree_MDI(
        q_value=0.05, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_01 = module_to_degree_MDI(
        q_value=0.01, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_005 = module_to_degree_MDI(
        q_value=0.005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_001 = module_to_degree_MDI(
        q_value=0.001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0005 = module_to_degree_MDI(
        q_value=0.0005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0001 = module_to_degree_MDI(
        q_value=0.0001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_00005 = module_to_degree_MDI(
        q_value=0.00005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_00001 = module_to_degree_MDI(
        q_value=0.00001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_000005 = module_to_degree_MDI(
        q_value=0.000005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_000001 = module_to_degree_MDI(
        q_value=0.000001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0000005 = module_to_degree_MDI(
        q_value=0.0000005, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)
    degree_MDI_0000001 = module_to_degree_MDI(
        q_value=0.0000001, module_4S=module_4S, module_I=module_I, module_III=module_III, module_IV=module_IV)

    # the differences of indices comparing with the first index
    variation_05 = sum(np.abs(np.subtract(degree_MDI_05, degree_MDI_05)))
    variation_01 = sum(np.abs(np.subtract(degree_MDI_01, degree_MDI_05)))
    variation_005 = sum(np.abs(np.subtract(degree_MDI_005, degree_MDI_01)))
    variation_001 = sum(np.abs(np.subtract(degree_MDI_001, degree_MDI_005)))
    variation_0005 = sum(np.abs(np.subtract(degree_MDI_0005, degree_MDI_001)))
    variation_0001 = sum(np.abs(np.subtract(degree_MDI_0001, degree_MDI_0005)))
    variation_00005 = sum(
        np.abs(np.subtract(degree_MDI_00005, degree_MDI_0001)))
    variation_00001 = sum(
        np.abs(np.subtract(degree_MDI_00001, degree_MDI_00005)))
    variation_000005 = sum(
        np.abs(np.subtract(degree_MDI_000005, degree_MDI_00001)))
    variation_000001 = sum(
        np.abs(np.subtract(degree_MDI_000001, degree_MDI_000005)))
    variation_0000005 = sum(
        np.abs(np.subtract(degree_MDI_0000005, degree_MDI_000001)))
    variation_0000001 = sum(
        np.abs(np.subtract(degree_MDI_0000001, degree_MDI_0000005)))

    variations = [variation_05, variation_01, variation_005, variation_001,
                  variation_0005, variation_0001, variation_00005,
                  variation_00001, variation_000005, variation_000001,
                  variation_0000005, variation_0000001]

    def variations_to_txt(path, variations):
        with open(path, 'w') as file:
            for i in variations:
                file.write(str(i) + "\n")

    variations_to_txt(path=path, variations=variations)

    return variations


def read_DEGs(path):
    import csv
    # Assuming a single column in the CSV file
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        csv_list = list(csv_reader)
        # print(csv_list)

    # Optionally, flatten the list if you have a single column
    csv_list = [i[0] for i in csv_list]
    csv_list.pop(0)

    return csv_list


def file_to_float(file):
    with open(file, 'r') as f:
        return [float(i.strip()) for i in f]


def file_to_string(file):
    with open(file, 'r') as f:
        return [i.strip() for i in f]


def find_targets(DEG, transitional_set):
    targets = list()
    for i in DEG:
        if i in transitional_set:
            targets.append(i)
    return targets


def targets_to_modules(target_genes, targets, first_module, second_module, first_module_stage, title):
    graph = nx.Graph()

    targets = target_genes[targets]
    for gene in targets:
        for i in first_module:
            if gene in first_module[i]:
                start = i
                break
        for j in second_module:
            if gene in second_module[j]:
                end = j
                break
        if graph.has_edge(i, j):
            graph[i][j]["gene"] += ", " + gene
        else:
            graph.add_edge(i, j, gene=gene)

    for i, j in graph.edges:
        if graph[i][j]["gene"].count(", ") >= 5:
            index = -1
            for k in range(5):
                index = graph[i][j]["gene"].find(", ", index+1)
            graph[i][j]["gene"] = graph[i][j]["gene"][:index+1] + \
                '\n' + graph[i][j]["gene"][index+2:]
        # print(f"source module: {i} -- {gene} -- target module: {j}")

    # # Print edges with attributes
    # for edge in graph.edges(data=True):
    #     print(edge)

    # Generate the bipartite layout
    pos = nx.bipartite_layout(graph, nodes=[i for i in graph.nodes if i.split("_")[
                              1] == first_module_stage])  # Specify the nodes of one bipartite set (optional)
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=[
                           node for node in graph.nodes()], node_color='skyblue', node_size=2000)
    # nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in graph.nodes() if isinstance(node, str)], node_color='b', node_size=500)

    # Add node labels
    nx.draw_networkx_labels(graph, pos, labels={node: node.split(
        "_")[1] + "_" + node.split("_")[2] for node in graph.nodes()})

    # Draw edges
    nx.draw_networkx_edges(graph, pos)

    # edge_labels = nx.get_edge_attributes(graph, 'gene')
    # nx.draw_networkx_edge_labels(
    #     graph, pos, edge_labels=edge_labels, font_size=10)

    # Show the plot
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(f"./figures/{title}.svg")
    plt.show()

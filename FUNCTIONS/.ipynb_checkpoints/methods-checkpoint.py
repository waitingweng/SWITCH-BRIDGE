import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from igraph import Graph, plot
import matplotlib.pyplot as plt
import leidenalg
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import json

# knn methods


def leiden_to_module(data, min_module_size=4):
    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for gene in data.obs_names:
        module_name = "module_" + \
            data.name.split("_")[-1] + "_" + data.obs.leiden.loc[gene]
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def module_to_json(module_dict, file_path):
    # Convert sets to lists in your dictionary
    module_dict_serializable = {
        key: list(value) for key, value in module_dict.items()
    }

    # Save the modified dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(module_dict_serializable, json_file)


def pcc_to_leiden(data, data_name, n_neighbors, resolution, fig_path):
    sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep='X')
    # Assuming Graph is from igraph library
    graph = Graph.Weighted_Adjacency(
        data.obsp["connectivities"], mode="lower", loops=False)
    print(graph.summary())

    # plot the degree distribution
    degree_sequence = graph.degree()

    # Set font sizes using rcParams
    # plt.rcParams.update({
    #     'axes.titlesize': 28,
    #     'axes.labelsize': 25,
    #     'xtick.labelsize': 20,
    #     'ytick.labelsize': 20,
    #     'legend.fontsize': 20
    # })
    # plt.figure(figsize=(10, 6))
    # plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
    #          density=True, alpha=0.7, color='b', edgecolor='black')
    # Add a vertical dashed line at x=30
    # plt.axvline(x=n_neighbors-1, color='r', linestyle='--', linewidth=2)
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.title('Degree Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path)
    # plt.show()

    sc.tl.leiden(data, resolution=resolution)
    data.write(data_name + ".h5ad")


def Binary_SWIM_pcc_to_distance(pcc_matrix, fig_path1, fig_path2, pcc_threshold=0.7):
    # construct a graph by PCC matrix, and then calculate pair-wise distances
    # Threshold for including edges
    gene_symbols = pcc_matrix.index
    # Convert PCC matrix to binary adjacency matrix
    adjacency_matrix = (abs(pcc_matrix) > pcc_threshold).astype(int)
    # Create an unweighted graph with gene names as vertices
    graph = Graph.Adjacency(adjacency_matrix, mode="UNDIRECTED", loops=False)
    graph.vs["name"] = gene_symbols
    # Display the graph summary
    print(graph.summary())
    # see whether the graph is connected (have to be connected)
    print(f"graph is connected: {graph.is_connected()}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()

    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    # Plot the degree distribution
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # Compute all pairs of nodes' distances
    all_pairs_distances = graph.distances()
    # Display the result
    print("All Pairs Distances:")
    print(type(all_pairs_distances))
    print(len(all_pairs_distances), len(all_pairs_distances[0]), sep='X')

    # Flatten the distances matrix
    # all_distances = [
    #     distance for distances_row in all_pairs_distances for distance in distances_row if distance != float('inf')]

    # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=np.arange(0, max(all_distances) + 0.1,
    #          0.1), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path3)
    # plt.show()

    all_pairs_distances = pd.DataFrame(
        all_pairs_distances, index=gene_symbols, columns=gene_symbols)

    # Extract the largest connected component (LCC) graph directly
    largest_connected_component = graph.components(mode='weak').giant()

    # Optional: Print information about the largest connected component (LCC) graph
    print(
        f"Number of vertices in largest connected component: {largest_connected_component.vcount()}")
    print(
        f"Number of edges in largest connected component: {largest_connected_component.ecount()}")
    largest_connected_component_genes = largest_connected_component.vs["name"]
    all_pairs_distances = all_pairs_distances.loc[largest_connected_component_genes,
                                                  largest_connected_component_genes]

    return all_pairs_distances


def Linear_SWIM_pcc_to_distance(pcc_matrix, fig_path1, fig_path2, fig_path3, pcc_threshold=0.7):
    # Get gene symbols as vertices
    gene_symbols = pcc_matrix.index
    # Create a weighted adjacency matrix based on the absolute value of PCC
    adjacency_matrix = 1 - np.abs(pcc_matrix.values)
    # Set weights to 0 for abs(PCC) <= 0.5
    adjacency_matrix[abs(pcc_matrix.values) <= pcc_threshold] = 0
    # Create a weighted graph using Weighted_Adjacency
    graph = Graph.Weighted_Adjacency(
        adjacency_matrix, mode="UNDIRECTED", attr="weight", loops=False)
    # Set vertex names
    graph.vs["name"] = gene_symbols
    # Display the graph summary
    print(graph.summary())
    # see whether the graph is connected (have to be connected)
    print(f"graph is connected: {graph.is_connected()}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()

    # Plot the degree distribution
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # Compute all pairs of nodes' distances
    all_pairs_distances = graph.distances(weights="weight")

    # Display the result
    print("All Pairs Distances:")
    print(type(all_pairs_distances))
    print(len(all_pairs_distances), len(all_pairs_distances[0]), sep='X')

    # Flatten the distances matrix
    # all_distances = [
    #     distance for distances_row in all_pairs_distances for distance in distances_row if distance != float('inf')]

    # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=np.arange(0, max(all_distances) + 0.1,
    #          0.1), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path3)
    # plt.show()

    all_pairs_distances = pd.DataFrame(
        all_pairs_distances, index=gene_symbols, columns=gene_symbols)

    # Extract the largest connected component (LCC) graph directly
    largest_connected_component = graph.components(mode='weak').giant()

    # Optional: Print information about the largest connected component (LCC) graph
    print(
        f"Number of vertices in largest connected component: {largest_connected_component.vcount()}")
    print(
        f"Number of edges in largest connected component: {largest_connected_component.ecount()}")
    largest_connected_component_genes = largest_connected_component.vs["name"]
    all_pairs_distances = all_pairs_distances.loc[largest_connected_component_genes,
                                                  largest_connected_component_genes]

    return all_pairs_distances


def Nlog_SWIM_pcc_to_distance(pcc_matrix, fig_path1, fig_path2, fig_path3, pcc_threshold=0.7):
    # Get gene symbols as vertices
    gene_symbols = pcc_matrix.index

    # 0 ~ 1
    abs_pcc = np.abs(pcc_matrix.values)
    # normalize the range of interest to 0 ~ 1
    normalized_pcc = (1 / (1-pcc_threshold)) * (abs_pcc - pcc_threshold)
    # set all negative values to 0
    non_negative_normalized_pcc = np.where(
        normalized_pcc < 0, 0, normalized_pcc)
    nlog = -np.log(non_negative_normalized_pcc)

    # set inf to 0
    adjacency_matrix = np.nan_to_num(nlog, posinf=0)

    # Create a weighted graph using Weighted_Adjacency
    graph = Graph.Weighted_Adjacency(
        adjacency_matrix, mode="UNDIRECTED", attr="weight", loops=False)

    # Set vertex names
    graph.vs["name"] = gene_symbols

    # Display the graph summary
    print(graph.summary())

    # see whether the graph is connected (have to be connected)
    print(f"graph is connected: {graph.is_connected()}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    # degree_sequence = graph.degree()

    # Plot the degree distribution
    # Set font sizes using rcParams
    # plt.rcParams.update({
    #     'axes.titlesize': 28,
    #     'axes.labelsize': 25,
    #     'xtick.labelsize': 20,
    #     'ytick.labelsize': 20,
    #     'legend.fontsize': 20
    # })
    # plt.figure(figsize=(10, 6))
    # plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
    #          density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.title('Degree Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path1)
    # plt.show()

    # Compute all pairs of nodes' distances
    all_pairs_distances = graph.distances(weights="weight")

    # Display the result
    print("All Pairs Distances:")
    print(type(all_pairs_distances))
    print(len(all_pairs_distances), len(all_pairs_distances[0]), sep='X')

    # Flatten the distances matrix
    # all_distances = [
    #     distance for distances_row in all_pairs_distances for distance in distances_row if distance != float('inf')]

    # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=np.arange(0, max(all_distances) + 0.1,
    #          0.1), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True, alpha=0.7, color='b', edgecolor='black')
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path3)
    # plt.show()

    all_pairs_distances = pd.DataFrame(
        all_pairs_distances, index=gene_symbols, columns=gene_symbols)

    # Extract the largest connected component (LCC) graph directly
    largest_connected_component = graph.components(mode='weak').giant()

    # Optional: Print information about the largest connected component (LCC) graph
    print(
        f"Number of vertices in largest connected component: {largest_connected_component.vcount()}")
    print(
        f"Number of edges in largest connected component: {largest_connected_component.ecount()}")
    largest_connected_component_genes = largest_connected_component.vs["name"]
    all_pairs_distances = all_pairs_distances.loc[largest_connected_component_genes,
                                                  largest_connected_component_genes]

    return all_pairs_distances


def Reciprocal_SWIM_pcc_to_distance(pcc_matrix, fig_path1, fig_path2, fig_path3, pcc_threshold=0.7):
    # Get gene symbols as vertices
    gene_symbols = pcc_matrix.index

    # 0 ~ 1
    abs_pcc = np.abs(pcc_matrix.values)
    # normalize the range of interest to 0 ~ 1
    normalized_pcc = (1 / (1-pcc_threshold)) * (abs_pcc - pcc_threshold)
    # set all negative values to 0
    non_negative_normalized_pcc = np.where(
        normalized_pcc < 0, 0, normalized_pcc)
    reciprocal = 1 / non_negative_normalized_pcc
    # set inf to 0
    adjacency_matrix = np.nan_to_num(reciprocal, posinf=0)

    # Create a weighted graph using Weighted_Adjacency
    graph = Graph.Weighted_Adjacency(
        adjacency_matrix, mode="UNDIRECTED", attr="weight", loops=False)

    # Set vertex names
    graph.vs["name"] = gene_symbols

    # Display the graph summary
    print(graph.summary())

    # see whether the graph is connected (have to be connected)
    print(f"graph is connected: {graph.is_connected()}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    # degree_sequence = graph.degree()

    # Plot the degree distribution
    # Set font sizes using rcParams
    # plt.rcParams.update({
    #     'axes.titlesize': 28,
    #     'axes.labelsize': 25,
    #     'xtick.labelsize': 20,
    #     'ytick.labelsize': 20,
    #     'legend.fontsize': 20
    # })
    # plt.figure(figsize=(10, 6))
    # plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
    #          density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.title('Degree Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path1)
    # plt.show()

    # Compute all pairs of nodes' distances
    all_pairs_distances = graph.distances(weights="weight")

    # Display the result
    print("All Pairs Distances:")
    print(type(all_pairs_distances))
    print(len(all_pairs_distances), len(all_pairs_distances[0]), sep='X')

    # Flatten the distances matrix
    # all_distances = [
    #     distance for distances_row in all_pairs_distances for distance in distances_row if distance != float('inf')]

    # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=np.arange(0, max(all_distances) + 0.1,
    #          0.1), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path3)
    # plt.show()

    all_pairs_distances = pd.DataFrame(
        all_pairs_distances, index=gene_symbols, columns=gene_symbols)

    # Extract the largest connected component (LCC) graph directly
    largest_connected_component = graph.components(mode='weak').giant()

    # Optional: Print information about the largest connected component (LCC) graph
    print(
        f"Number of vertices in largest connected component: {largest_connected_component.vcount()}")
    print(
        f"Number of edges in largest connected component: {largest_connected_component.ecount()}")
    largest_connected_component_genes = largest_connected_component.vs["name"]
    all_pairs_distances = all_pairs_distances.loc[largest_connected_component_genes,
                                                  largest_connected_component_genes]

    return all_pairs_distances


def distance_to_leiden(data, data_name, n_neighbors, resolution, fig_path):
    sc.pp.neighbors(data, n_neighbors=n_neighbors, use_rep='X')
    # Assuming Graph is from igraph library
    graph = Graph.Weighted_Adjacency(
        data.obsp["connectivities"], mode="lower", loops=False)
    print(graph.summary())

    # plot the degree distribution
    degree_sequence = graph.degree()
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    # Add a vertical dashed line at x=30
    plt.axvline(x=n_neighbors-1, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path)
    plt.show()

    sc.tl.leiden(data, resolution=resolution)
    data.write(data_name + ".h5ad")


def module_to_dict(file=""):
    # importing the module
    # Opening JSON file
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        return data


def true_module_to_dict(file=""):
    # importing the module
    # Opening JSON file
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        return {key: value for key, value in data.items() if len(value) >= 4}


def module_info(module):
    total_modules = 0
    modules_3 = 0
    modules_1000 = 0
    total_genes = 0
    genes_3 = 0
    genes_1000 = 0
    for i in module:
        total_modules += 1
        total_genes += len(module[i])
        if len(module[i]) <= 3:
            modules_3 += 1
            genes_3 += len(module[i])
        elif len(module[i]) >= 1000:
            modules_1000 += 1
            genes_1000 += len(module[i])

    print(
        f"total modules = {total_modules}, modules <= 3 = {modules_3}, modules >= 1000 = {modules_1000}")
    print(f"modules <= 3 ratio = {modules_3/total_modules}, modules >= 1000 ratio = {modules_1000/total_modules}, other modules ratio = {(total_modules-modules_3-modules_1000)/total_modules}")
    print(
        f"total genes = {total_genes}, genes <= 3 = {genes_3}, genes >= 1000 = {genes_1000}")
    print(f"genes <= 3 ratio = {genes_3/total_genes}, genes >= 1000 ratio = {genes_1000/total_genes}, other genes ratio = {(total_genes-genes_3-genes_1000)/total_genes}")
    print()


def get_distribution(data):
    # List of distributions to test
    distributions = [
        'weibull_min',  # Weibull distribution (minimum version)
        'gamma',        # Gamma distribution
        'lognorm',      # Log-Normal distribution
        'invgauss',
        'genpareto',
        'powerlaw'
    ]

    # Initialize lists to store test results
    test_results = []

    # Fit each distribution and perform KS test
    for dist_name in distributions:
        # Fit distribution to data
        dist = getattr(stats, dist_name)
        params = dist.fit(data)

        # Perform KS test
        ks_stat, p_value = stats.kstest(data, dist_name, args=params)

        # Store test results
        test_results.append((dist_name, ks_stat, p_value, params))

        # Plot fitted distribution against data (optional)
        plt.figure()
        plt.hist(data, bins=30, density=True, alpha=0.5,
                 label='Data')  # Histogram of data
        x = np.linspace(min(data), max(data), 100)
        plt.plot(x, dist.pdf(x, *params), 'r-', lw=2,
                 label=dist_name)  # Fitted distribution
        plt.title(f'{dist_name} Fit to Data')
        plt.legend()
        plt.show()

    # Print test results
    for result in test_results:
        dist_name, ks_stat, p_value, params = result
        print(
            f'{dist_name}: KS Statistic = {ks_stat:.4f}, p-value = {p_value:.4f}, Params = {params}')


def module_size_distribution(module_dict):
    # Extract module sizes from the dictionary
    module_sizes = [len(module) for module in module_dict.values()]
    get_distribution(data=module_sizes)
    # Plot histogram of module sizes
    plt.hist(module_sizes, bins=range(min(module_sizes),
             max(module_sizes) + 1), edgecolor='black')
    plt.xlabel('Module Size')
    plt.ylabel('Frequency')
    plt.title('Distribution of Module Sizes')
    plt.grid(True)
    plt.show()


# distance-based method
def anndata_to_transposed_dataframe(adata):
    # Create a DataFrame and transpose it
    df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # Display the DataFrame
    # print(df.shape)
    df = df.T
    print(df.shape)
    # print(df)
    return df


def raw_to_pca(raw_matrix, num_pcs_to_preserve, file):
    # Assuming 'data' is your 12850-dimensional dataset
    # Scales it to have zero mean and unit variance for each feature
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(raw_matrix)

    # Apply PCA
    pca = PCA()
    pca.fit(data_standardized)

    # Plot the cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(cumulative_explained_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    # plt.xlim(0, 100)
    plt.show()

    # Apply PCA with the chosen number of PCs
    pca = PCA(n_components=num_pcs_to_preserve)
    data_pca = pca.fit_transform(data_standardized)
    data_pca = pd.DataFrame(data_pca)
    data_pca.index = raw_matrix.index
    data_pca.to_csv(file)
    return data_pca


def pcc_to_pca(pcc_matrix, num_pcs_to_preserve, file):
    # Assuming 'data' is your 12850-dimensional dataset
    # Scales it to have zero mean and unit variance for each feature
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(pcc_matrix)

    # Apply PCA
    pca = PCA()
    pca.fit(data_standardized)

    # Plot the cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(cumulative_explained_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    # plt.xlim(0, 100)
    plt.show()

    # Apply PCA with the chosen number of PCs
    pca = PCA(n_components=num_pcs_to_preserve)
    data_pca = pca.fit_transform(data_standardized)
    data_pca = pd.DataFrame(data_pca)
    data_pca.index = pcc_matrix.index
    data_pca.to_csv(file)
    return data_pca


def pca_to_leiden(pca_coordinate, threshold_percentile=50, resolution_parameter=0.01, seed=123456):
    # generate a graph with the pca_coordinate
    # Assuming your gene data is stored in 'gene_pca'
    # Replace this with your actual PCA-transformed gene data
    # gene_pca = load_your_pca_transformed_gene_data_function()

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(pca_coordinate, metric='euclidean')

    # Find the shortest 10% of distances
    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Select only the distances below the threshold
    selected_distances = distances[distances < threshold_distance]

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (selected_distances - selected_distances.min()) / \
        (selected_distances.max() - selected_distances.min())

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        (distances < threshold_distance).astype(int).tolist(), mode="UNDIRECTED")

    # Set edge weights for selected distances after creating the graph
    selected_edges = list(zip(*np.where(distances < threshold_distance)))

    for edge, weight in zip(selected_edges, 1 / (normalized_distances + 1e-10)):
        # Set the weight for each selected edge
        graph.es[edge]["weight"] = weight

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()

    # Plot the degree distribution
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph

    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership)

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pca_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pca_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # # use the cluster_dict as needed
    # print(len(module_dict))
    return module_dict


def raw_to_module(raw_coordinate, threshold_percentile=5, resolution_parameter=0.5, seed=123456):
    # generate a graph with the pcc_coordinate
    # Assuming your gene data is stored in 'gene_pca'
    # Replace this with your actual PCA-transformed gene data
    # gene_pca = load_your_pca_transformed_gene_data_function()

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(raw_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Select only the distances below the threshold
    selected_distances = distances[distances < threshold_distance]

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (selected_distances - selected_distances.min()) / \
        (selected_distances.max() - selected_distances.min())

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency((distances < threshold_distance).astype(
        int).tolist(), mode="UNDIRECTED", loops=False)

    # Set edge weights for selected distances after creating the graph
    selected_edges = list(zip(*np.where(distances < threshold_distance)))

    for edge, weight in zip(selected_edges, 1 / (normalized_distances + 1e-10)):
        # Set the weight for each selected edge
        graph.es[edge]["weight"] = weight
    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = raw_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            raw_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # use the cluster_dict as needed
    # print(len(module_dict))
    # print(list(module_dict.keys())[:20])
    return module_dict


def PCC_SWIM_pcc_to_module_binary(pcc_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(pcc_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        (distances < threshold_distance).astype(int), mode="UNDIRECTED", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weight_distribution, bins=10000, density=True,
             alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    # Save the plot to a file
    plt.savefig(fig_path2)
    plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pcc_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pcc_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def PCC_SWIM_pcc_to_module_reciprocal(pcc_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(pcc_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # linear y = 1/x transform
    transformed_distances = (
        1 / (normalized_distances + 1e-10)) - (1 / (1 + 1e-10))

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weight_distribution, bins=10000, density=True,
             alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    # Save the plot to a file
    plt.savefig(fig_path2)
    plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pcc_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pcc_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def PCC_SWIM_pcc_to_module_nlog(pcc_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(pcc_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # y = -ln(x) transform
    transformed_distances = - \
        np.log(normalized_distances + 1e-10) + np.log(1 + 1e-10)

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weight_distribution, bins=10000, density=True,
             alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    # Save the plot to a file
    plt.savefig(fig_path2)
    plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pcc_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pcc_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def PCC_SWIM_pcc_to_module_linear(pcc_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(pcc_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # linear y = 1-x transform
    transformed_distances = 1 - normalized_distances

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    # Set font sizes using rcParams
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    ####### Cannot run properly #######

    # weight_distribution = graph.es["weight"]

    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #         alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    ###################################

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pcc_coordinate.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pcc_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def distance_to_module_binary(distance_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(distance_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        (distances < threshold_distance).astype(int), mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    # Replace with your actual gene symbols
    gene_symbols = distance_coordinate.index

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            distance_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def distance_to_module_nlog(distance_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(distance_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # linear y = 1/x transform
    transformed_distances = - \
        np.log(normalized_distances + 1e-10) + np.log(1 + 1e-10)

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    # Replace with your actual gene symbols
    gene_symbols = distance_coordinate.index

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            distance_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def distance_to_module_reciprocal(distance_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(distance_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # linear y = 1/x transform
    transformed_distances = (
        1 / (normalized_distances + 1e-10)) - (1 / (1 + 1e-10))

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weight_distribution, bins=10000, density=True,
             alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    # Save the plot to a file
    plt.savefig(fig_path2)
    plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    # Replace with your actual gene symbols
    gene_symbols = distance_coordinate.index

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            distance_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def distance_to_module_linear(distance_coordinate, fig_path1, fig_path2, threshold_percentile=10, resolution_parameter=0.5, min_module_size=4, seed=123456):

    # Calculate pairwise Euclidean distances
    distances = pairwise_distances(distance_coordinate, metric='euclidean')

    threshold_distance = np.percentile(distances, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_distance = {threshold_distance}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    distances[distances > threshold_distance] = threshold_distance

    # Normalize only the selected distances to [0, 1]
    normalized_distances = (distances - distances.min()) / \
        (distances.max() - distances.min())

    # linear y = 1-x transform
    transformed_distances = 1 - normalized_distances

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        transformed_distances, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    # Replace with your actual gene symbols
    gene_symbols = distance_coordinate.index

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            distance_coordinate.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


# method7
def Leiden_pcc_to_module(pcc_matrix, fig_path1, fig_path2, fig_path3, pcc_threshold=0.5, resolution_parameter=0.35, min_module_size=4, seed=123456):
    # Threshold for including edges
    # Get gene symbols as vertices
    gene_symbols = pcc_matrix.index

    # Create a weighted adjacency matrix based on the absolute value of PCC
    adjacency_matrix = abs(pcc_matrix.values)
    # Set weights to 0 for abs(PCC) <= 0.5
    adjacency_matrix[adjacency_matrix <= pcc_threshold] = 0

    # Create a weighted graph using Weighted_Adjacency
    graph = Graph.Weighted_Adjacency(
        adjacency_matrix, mode="UNDIRECTED", attr="weight", loops=False)
    # Set vertex names
    graph.vs["name"] = gene_symbols
    # Display the graph summary
    print(graph.summary())
    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()

    # Plot the degree distribution
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # plot the shortest_distances' distribution
    # Get all pairs of nodes' distances
    distances = graph.distances()

    # Flatten the distances matrix
    all_distances = [
        distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2),
    #          density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    # weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(weight_distribution, bins=10000, density=True,
    #          alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    # plt.title('Weight Distribution')
    # Save the plot to a file
    # plt.savefig(fig_path3)
    # plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph

    # Set the number of iterations
    n_iterations = -1
    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    # print("Cluster assignments:", partition.membership[:10])

    # # plot the leiden clusters
    # # Convert layout to NumPy array
    # layout = np.array(graph.layout("kk"))

    # # Visualize the graph with color-coded clusters using matplotlib
    # plt.figure(figsize=(10, 8))
    # plt.scatter(layout[:, 0], layout[:, 1], c=partition.membership, cmap='viridis', s=1, alpha=0.8)
    # plt.title('Leiden Clustering Results')
    # plt.show()

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    gene_symbols = pcc_matrix.index  # Replace with your actual gene symbols

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            pcc_matrix.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


# hdWGCNA
def hdWGCNA_to_module(stage, path):
    module_table = pd.read_csv(path, index_col=0)

    # Initialize an empty dictionary
    module_dict = {}

    for gene_symbol in module_table.index:
        module_name = module_table.loc[gene_symbol, "module"]

        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(f"module_{stage}_" + module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[f"module_{stage}_" + module_name].add(gene_symbol)
    print(len(module_dict))

    return module_dict


def TOM_to_module(TOM, fig_path1, fig_path2, threshold_percentile=50, resolution_parameter=0.5, min_module_size=4, seed=123456):

    TOM_array = TOM.to_numpy()
    threshold_weight = np.percentile(TOM_array, threshold_percentile)
    print(
        f"threshold_percentile = {threshold_percentile}%: threshold_weight = {threshold_weight}")

    # Create a weighted adjacency matrix based on the absolute value of PCC
    TOM_array[TOM_array < threshold_weight] = 0

    # Create an igraph graph and add edges based on the threshold
    graph = Graph.Weighted_Adjacency(
        TOM_array, mode="lower", loops=False)

    # Get the connected components
    components = graph.connected_components()
    # Get the number of connected components
    num_components = components.__len__()
    # Print the result
    print(f"Number of connected components: {num_components}")

    # plot the degree distribution
    # Get the degree sequence of the graph
    degree_sequence = graph.degree()
    # Plot the degree distribution
    plt.rcParams.update({
        'axes.titlesize': 28,
        'axes.labelsize': 25,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(max(degree_sequence) + 2),
             density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    # Save the plot to a file
    plt.savefig(fig_path1)
    plt.show()

    # # plot the shortest_distances' distribution
    # # Get all pairs of nodes' distances
    # distances = graph.distances()

    # # Flatten the distances matrix
    # all_distances = [distance for distances_row in distances for distance in distances_row if distance != float('inf')]

    # # Plot the distance distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(all_distances, bins=range(int(max(all_distances)) + 2), density=True, alpha=0.7, color='b', edgecolor='black')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title('Distance Distribution')
    # # Save the plot to a file
    # plt.savefig(fig_path2)
    # plt.show()

    weight_distribution = graph.es["weight"]
    # Plot the weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weight_distribution, bins=10000, density=True,
             alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    # Save the plot to a file
    plt.savefig(fig_path2)
    plt.show()

    # calculate the leiden clusters
    # Assuming 'graph' is your igraph graph
    # Set the number of iterations
    n_iterations = -1

    partition = leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights="weight",
                                         resolution_parameter=resolution_parameter, n_iterations=n_iterations, seed=seed)
    # Assess or visualize the results based on the current resolution_parameter
    # Continue with additional analysis or choose the best resolution based on your criteria

    # Additional steps: Print or visualize clustering results
    num_clusters = max(partition.membership) + 1
    print(f"Number of clusters: {num_clusters}")
    print("Cluster assignments:", partition.membership[:20])

    # Assuming 'gene_symbols' is a list of gene symbols corresponding to the graph nodes
    # Replace with your actual gene symbols
    gene_symbols = TOM.index

    # Initialize an empty dictionary
    module_dict = {}

    # Iterate through nodes and their Leiden cluster IDs
    for node, module_id in enumerate(partition.membership):
        # Get the gene symbol corresponding to the current node
        gene_symbol = gene_symbols[node]

        module_name = "module_" + \
            TOM.name.split("_")[-1] + "_" + str(module_id)
        # print(pcc_matrix.name)
        # print(module_name)
        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(module_name, set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[module_name].add(gene_symbol)

    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict


def hotspot_to_module(stage, modules):
    # Initialize an empty dictionary
    module_dict = {}

    for gene_symbol in modules.index:
        module_name = modules[gene_symbol]

        if module_name != -1:
            # If the cluster ID is not in the dictionary, add it with an empty set
            module_dict.setdefault(f"module_{stage}_{module_name}", set())

            # Add the gene symbol to the set of the corresponding module ID
            module_dict[f"module_{stage}_{module_name}"].add(gene_symbol)
    print(len(module_dict))

    return module_dict


def NMF_to_module(stage, modules, genes, min_module_size=4):
    # Initialize an empty dictionary
    module_dict = {}
    gene_assignments = np.argmax(modules, axis=0)
    # print(len(set(gene_assignments)))
    for i in range(modules.shape[1]):
        gene_symbol = genes[i]
        module_name = gene_assignments[i]

        # If the cluster ID is not in the dictionary, add it with an empty set
        module_dict.setdefault(f"module_{stage}_{module_name}", set())

        # Add the gene symbol to the set of the corresponding module ID
        module_dict[f"module_{stage}_{module_name}"].add(gene_symbol)

    print(len(module_dict))
    # Remove key-value pairs where the list length is less than 4
    filtered_module_dict = {
        k: v for k, v in module_dict.items() if len(v) >= min_module_size}
    print(f"Number of modules: {len(filtered_module_dict)}")

    return filtered_module_dict

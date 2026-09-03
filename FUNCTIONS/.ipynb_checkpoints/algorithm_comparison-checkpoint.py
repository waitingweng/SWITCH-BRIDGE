import gseapy
import numpy as np
import json
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def enrich_single_module(args):
    module_id, gene_list, gene_sets, min_module_size = args
    if len(gene_list) < min_module_size:
        return (module_id, False, 0)

    try:
        results = gseapy.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            background="background.txt",
            outdir=None
        )
        if results.res2d['Adjusted P-value'].iloc[0] < 0.05:
            return (module_id, True, len(gene_list))
    except:
        pass
    
    return (module_id, False, 0)

def enrichment(modules, gene_sets, min_module_size, total_genes):
    args = [(i, modules[i], gene_sets, min_module_size) for i in modules]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(enrich_single_module, args)

    enrichment_dict = {}
    pathways = 0
    correctly_assigned_genes = 0
    total_modules = 0

    for module_id, enriched, gene_count in results:
        if len(modules[module_id]) >= min_module_size:
            total_modules += 1
            enrichment_dict[module_id] = enriched
            if enriched:
                pathways += 1
                correctly_assigned_genes += gene_count

    enrichment_ratio = pathways / total_modules if total_modules else 0
    correctly_assigned_ratio = correctly_assigned_genes / total_genes if total_genes else 0

    print(
        f"pathways: {pathways}, total modules: {total_modules}, enrichment_ratio: {enrichment_ratio}")
    print(
        f"correctly_assigned_genes: {correctly_assigned_genes}, total genes: {total_genes}, correctly_assigned_ratio: {correctly_assigned_ratio}")

    return enrichment_ratio, correctly_assigned_ratio


# def enrichment(modules, gene_sets, min_module_size, total_genes):
#     pathways = 0
#     total_modules = 0
#     enrichment_dict = {}

#     for i in modules:
#         if len(modules[i]) >= min_module_size:
#             total_modules += 1
#             enrichment_dict[i] = False
#             gl = modules[i]
#             results = gseapy.enrichr(
#                 gene_list=gl, gene_sets=gene_sets, background="background.txt", outdir=None)
#             try:
#                 if results.res2d['Adjusted P-value'].iloc[0] < 0.05:
#                     pathways += 1
#                     enrichment_dict[i] = True
#             except:
#                 pass

#     enrichment_ratio = pathways/total_modules
#     print(
#         f"pathways: {pathways}, total modules: {total_modules}, enrichment_ratio: {enrichment_ratio}")

#     correctly_assigned_genes = 0
#     for i in enrichment_dict:
#         if enrichment_dict[i] == True:
#             correctly_assigned_genes += len(modules[i])
#     correctly_assigned_ratio = correctly_assigned_genes/total_genes
#     print(
#         f"correctly_assigned_genes: {correctly_assigned_genes}, total genes: {total_genes}, correctly_assigned_ratio: {correctly_assigned_ratio}")

#     return enrichment_ratio, correctly_assigned_ratio


def list_to_txt(path, number_list):
    with open(path, 'w') as file:
        for i in number_list:
            file.write(str(i) + "\n")


def file_to_float(path):
    l = list()
    with open(path, "r") as file:
        for i in file:
            l.append(float(i.strip()))
    return l


def ttest(list1, list2):
    from scipy.stats import ttest_ind
    # Assuming sil_scores1 and sil_scores2 are your two sets of silhouette scores
    t_stat, p_value = ttest_ind(list1, list2)

    # Check if the difference is statistically significant
    if p_value < 0.05:
        print("The difference is statistically significant.")
    else:
        print("There is no significant difference.")
    print("p_value =", p_value)


def number_of_modules(modules, min_module_size=10):
    total_modules = 0
    for i in modules:
        if len(modules[i]) >= min_module_size:
            total_modules += 1
    return total_modules


def valid_ratio(modules, min_module_size=4):
    all_detected_modules = len(modules)
    valid_modules = 0
    for i in modules:
        if len(modules[i]) >= min_module_size:
            valid_modules += 1

    valid_ratio = valid_modules/all_detected_modules
    return valid_ratio


def module_ratio(modules, total_genes):
    module_genes = 0
    for key in modules:
        module_genes += len(modules[key])

    return module_genes/total_genes


def build_overlap_matrix(modules_before, modules_after):
    overlap_matrix = np.zeros(
        (len(modules_before), len(modules_after)), dtype=int)
    for i, key_before in enumerate(modules_before):
        for j, key_after in enumerate(modules_after):
            overlap_matrix[i, j] = len(
                set(modules_before[key_before]).intersection(modules_after[key_after]))
    return overlap_matrix


def find_best_matches(overlap_matrix):
    # Negate for maximization
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    return row_ind, col_ind


def calculate_reproducibility_score(modules_before, modules_after):
    overlap_matrix = build_overlap_matrix(modules_before, modules_after)
    row_ind, col_ind = find_best_matches(overlap_matrix)

    total_overlap = overlap_matrix[row_ind, col_ind].sum()
    max_possible_overlap = min(sum(len(modules_before[key]) for key in modules_before), sum(
        len(modules_after[key]) for key in modules_after))

    reproducibility_score = total_overlap / max_possible_overlap
    return reproducibility_score, row_ind, col_ind, overlap_matrix


def reproducibility_to_txt(resolutions, module_4S_list, module_I_list, module_III_list, module_IV_list, path):
    reproducibility = list()

    for count in range(len(resolutions)-1):
        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_4S_list[count], module_4S_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_I_list[count], module_I_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_III_list[count], module_III_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_IV_list[count], module_IV_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        # reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(module_NA_list[count], module_NA_list[count+1])
        # print(reproducibility_score)
        # reproducibility.append(reproducibility_score)

        print()

    list_to_txt(path, reproducibility)

def reproducibility_to_txt_COVID(resolutions, module_4S_list, module_I_list, module_III_list, path):
    reproducibility = list()

    for count in range(len(resolutions)-1):
        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_4S_list[count], module_4S_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_I_list[count], module_I_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(
            module_III_list[count], module_III_list[count+1])
        print(reproducibility_score)
        reproducibility.append(reproducibility_score)

        # reproducibility_score, row_ind, col_ind, overlap_matrix = calculate_reproducibility_score(module_NA_list[count], module_NA_list[count+1])
        # print(reproducibility_score)
        # reproducibility.append(reproducibility_score)

        print()

    list_to_txt(path, reproducibility)

def module_dicts_to_lists(modules_dict1, modules_dict2):
    # Step 1: Find the intersection of genes
    genes1 = set(gene for genes in modules_dict1.values() for gene in genes)
    genes2 = set(gene for genes in modules_dict2.values() for gene in genes)
    common_genes = genes1.intersection(genes2)

    # Step 2: Create cluster labels for the common genes
    def create_labels(modules_dict, common_genes):
        labels = {}
        for module_id, genes in modules_dict.items():
            for gene in genes:
                if gene in common_genes:
                    labels[gene] = module_id
        return labels

    labels1 = create_labels(modules_dict1, common_genes)
    labels2 = create_labels(modules_dict2, common_genes)

    # Convert labels to lists in the same order
    common_genes = sorted(common_genes)  # Ensure genes are in the same order
    labels_list1 = [labels1[gene] for gene in common_genes]
    labels_list2 = [labels2[gene] for gene in common_genes]

    return labels_list1, labels_list2


def dict_to_json(data, path):
    # Convert tuple keys to strings for JSON compatibility
    data_json_ready = {str(key): value for key, value in data.items()}

    # Save to a JSON file
    with open(path, 'w') as json_file:
        json.dump(data_json_ready, json_file)


def interstage_ARI(module_list, stage_name, path):
    ARI_dict = dict()
    for i in range(len(module_list)):
        for j in range(i+1, len(module_list)):
            l1, l2 = module_dicts_to_lists(module_list[i], module_list[j])

            ari = adjusted_rand_score(l1, l2)
            ARI_dict[(stage_name[i], stage_name[j])] = ari

    print(ARI_dict)
    dict_to_json(ARI_dict, path)


def interstage_AMI(module_list, stage_name, path):
    AMI_dict = dict()
    for i in range(len(module_list)):
        for j in range(i+1, len(module_list)):
            l1, l2 = module_dicts_to_lists(module_list[i], module_list[j])

            ami = adjusted_mutual_info_score(l1, l2)
            AMI_dict[(stage_name[i], stage_name[j])] = ami

    print(AMI_dict)
    dict_to_json(AMI_dict, path)


def intrastage_ARI(module_list):
    ARI = list()
    for i in range(len(module_list)-1):
        l1, l2 = module_dicts_to_lists(module_list[i], module_list[i+1])

        ari = adjusted_rand_score(l1, l2)
        ARI.append(ari)

    print(ARI)
    return ARI


def intrastage_AMI(module_list):
    AMI = list()
    for i in range(len(module_list)-1):
        l1, l2 = module_dicts_to_lists(module_list[i], module_list[i+1])

        ami = adjusted_mutual_info_score(l1, l2)
        AMI.append(ami)

    print(AMI)
    return AMI


def jaccard_index(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1 & set2)  # Intersection of sets
    union = len(set1 | set2)         # Union of sets
    return intersection / union


def relevance(my_modules, curated_modules):
    Jaccard_sum = 0
    for i in my_modules:
        max_j_id = 0
        for j in curated_modules:
            j_id = jaccard_index(my_modules[i], curated_modules[j])
            max_j_id = max(j_id, max_j_id)
        Jaccard_sum += max_j_id
    return Jaccard_sum / len(my_modules)

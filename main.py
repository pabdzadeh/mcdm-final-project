import networkx as nx
import numpy as np
import scipy
import pandas as pd
import itertools as IT
import random
from pyDecisions.py_decisions.topsis.topsis import topsis_method
from matplotlib import pyplot as plt
from colorama import Fore
from colorama import Style


def import_dataset(path, number_of_nodes, number_of_layers=3):
    edges = pd.read_csv(filepath_or_buffer=path, sep=',', header=0)
    graphs = []

    for i in range(number_of_layers):
        g = nx.from_pandas_edgelist(df=edges.loc[edges['layer'] == (i + 1)], source='source', target='target',
                                    create_using=nx.Graph())
        for i in range(number_of_nodes):
            if not g.has_node(i):
                g.add_node(i)
        graphs.append(g)
    return graphs


def find_not_existing_links(target_layer_graph):
    not_existing = [pair for pair in IT.combinations(target_layer_graph.nodes(), 2) if
                    not target_layer_graph.has_edge(*pair)]

    return not_existing


def set_missing_links(edges, ratio=0.1):
    return random.sample(edges, int(ratio * len(edges)))


def common_neighbor_similarity(graph, links):
    return [(link[0], link[1], len(list(nx.common_neighbors(graph, link[0], link[1])))) for link in links]


def calculate_node_similarities(graph, links, mode='Jaccard'):
    if mode == 'Jaccard':
        return nx.jaccard_coefficient(graph, links)
    elif mode == 'CN':
        return common_neighbor_similarity(graph, links)
    elif mode == 'RA':
        return nx.resource_allocation_index(graph, links)
    elif mode == 'AA':
        return nx.adamic_adar_index(graph, links)
    elif mode == 'PA':
        return nx.preferential_attachment(graph, links)
    return nx.jaccard_coefficient(graph, links)


def calculate_layer_similarity(layer_alpha, layer_beta, mode='OR'):
    if mode == 'OR':
        intersection = nx.operators.intersection(layer_alpha, layer_beta)
        return 2 * intersection.number_of_edges() / (layer_alpha.number_of_edges() + layer_beta.number_of_edges())
    elif mode == 'PCC':
        number_of_nodes_alpha = layer_alpha.number_of_nodes()
        adj_alpha = np.array(nx.adjacency_matrix(layer_alpha, nodelist=range(number_of_nodes_alpha)).todense())
        mean_alpha = adj_alpha.mean()
        std_alpha = adj_alpha.std()

        number_of_nodes_beta = layer_beta.number_of_nodes()
        adj_beta = np.array(nx.adjacency_matrix(layer_beta, nodelist=range(number_of_nodes_beta)).todense())
        mean_beta = adj_beta.mean()
        std_beta = adj_beta.std()

        pcc_result = 0
        for i in range(number_of_nodes_alpha):
            for j in range(number_of_nodes_beta):
                pcc_result += ((adj_alpha[i, j] - mean_alpha) * (adj_beta[i, j] - mean_beta) / (std_alpha * std_beta))

        pcc_result /= (number_of_nodes_alpha ** 2)
        return pcc_result
    elif mode == 'AASN':
        intersection = nx.operators.intersection(layer_alpha, layer_beta)
        return intersection.number_of_edges() / layer_alpha.number_of_edges()


def create_decision_matrix(graphs, possible_links, intra_layer_similarity='RA'):
    all_scores_per_layer = []

    for graph in graphs:
        layer_scores = [p for u, v, p in calculate_node_similarities(graph, links=possible_links,
                                                                     mode=intra_layer_similarity)]
        all_scores_per_layer.append(layer_scores)
    decision_matrix = np.array(all_scores_per_layer)
    return decision_matrix.T


def calculate_weights(graphs, target_layer, phi=0.8, inter_layer_similarity='PCC'):
    weights = []
    similarities = []
    for i, layer_graph in enumerate(graphs):
        if i == target_layer:
            continue
        similarities.append(calculate_layer_similarity(graphs[target_layer], layer_graph, mode=inter_layer_similarity))
    similarities = np.array(similarities)
    for i, graph in enumerate(graphs):
        if i == target_layer:
            weights.append(1 - phi)
        else:
            weight = phi * (calculate_layer_similarity(graphs[target_layer], graph, mode=inter_layer_similarity))
            weight = weight / similarities.sum()
            weights.append(weight)
    return weights


def evaluate_results(results, possible_links, missing_links, not_existing_links, method='AUC', k=10):
    results_per_link = {}
    for index, link in enumerate(possible_links):
        results_per_link[(link[0], link[1])] = results[index]

    if method == 'AUC':
        num_of_greater = 0
        num_of_equal = 0
        num_of_comparisons = 0
        for missing_link in missing_links:
            for not_existing_link in not_existing_links:
                if results_per_link[(missing_link[0], missing_link[1])] > results_per_link[(not_existing_link[0],
                                                                                            not_existing_link[1])]:
                    num_of_greater += 1
                elif results_per_link[(missing_link[0], missing_link[1])] == results_per_link[(not_existing_link[0],
                                                                                               not_existing_link[1])]:
                    num_of_equal += 1
                num_of_comparisons += 1
        return (num_of_greater + 0.5 * num_of_equal) / num_of_comparisons

    elif method == 'precision':
        result = 0
        for link in possible_links[results.argsort()[-k:][::-1]]:
            for missing_link in missing_links:
                if link[0] == missing_link[0] and link[1] == missing_link[1]:
                    result += 1
        return result / k
    return


def tune_phi(graphs, dataset, target_layer, possible_links, missing_links, not_existing_links, decision_matrix):
    print('Tuning Phi start')
    counter = 0
    results_per_phi = []
    phi_array = []

    while counter < 11:
        weights = calculate_weights(graphs, target_layer, phi=0.1 * counter)
        phi_array.append(0.1 * counter)
        topsis_results = topsis_method(decision_matrix, weights, ['max' for _ in graphs]).round(decimals=2)
        auc = evaluate_results(topsis_results, possible_links, missing_links, not_existing_links, method='AUC')
        results_per_phi.append(auc)
        counter += 1

    results_per_phi = np.array(results_per_phi)
    phi = phi_array[results_per_phi.argsort()[-1:][::-1][0]]
    plot(phi_array, results_per_phi, 'phi', 'AUC', name=dataset['name'] + '_layer_' + str(target_layer))
    print('Tuning Phi end')
    return phi


def run_one_iteration(run_number, dataset, graphs, config):
    not_existing_links = np.array(find_not_existing_links(graphs[config.target_layer]))
    missing_links = np.array(set_missing_links(graphs[config.target_layer].edges(), ratio=config.test_ratio))
    possible_links = np.concatenate((not_existing_links, missing_links))

    decision_matrix = create_decision_matrix(graphs, possible_links,
                                             intra_layer_similarity=config.intra_layer_similarity)

    phi = config.phi
    if run_number == 0 and config.use_tuned_phi:
        phi = round(tune_phi(graphs, dataset, config.target_layer, possible_links, missing_links, not_existing_links,
                             decision_matrix), 2)
        print('Phi = ', phi)

    weights = calculate_weights(graphs, config.target_layer, phi=phi)
    results = topsis_method(decision_matrix, weights, ['max' for _ in graphs]).round(decimals=2)

    precision = evaluate_results(results, possible_links, missing_links, not_existing_links,
                                 method='precision', k=config.k_for_precision)
    auc = evaluate_results(results, possible_links, missing_links, not_existing_links, method='AUC')

    print(f'#{run_number}, {dataset["name"]},  {config.inter_layer_similarity}, {config.intra_layer_similarity},'
          f' Precision: {round(precision, 4)}, AUC: {round(auc, 4)}')

    return precision, auc


def run_for_one_layer(dataset, graphs, config, target_layer):
    print(f'{Fore.RED}***************')
    print(f'Target Layer: {target_layer + 1}')
    print(f'***************{Fore.RESET}')

    precisions = []
    aucs = []
    config.target_layer = target_layer

    for i in range(config.runs):
        precision, auc = run_one_iteration(i, dataset, graphs, config)
        precisions.append(precision)
        aucs.append(auc)
    print(f'\n{Fore.GREEN}--------------------------------------------')
    print('Target Layer: ', target_layer + 1)
    print('Layer AVG Precision = ', round(np.array(precisions).mean(), 4))
    print('Layer AVG AUC = ', round(np.array(aucs).mean(), 4))
    print(f'--------------------------------------------\n{Fore.RESET}')
    return precisions, aucs


def run_for_a_dataset(dataset, config):
    dataset_aucs = []
    dataset_precisions = []
    print_dataset_name(dataset)
    graphs = import_dataset(path='./data/' + dataset["name"] + '/edges.csv',
                            number_of_nodes=dataset['number_of_nodes'], number_of_layers=dataset['layers'])

    for target_layer in range(dataset['layers']):
        precisions, aucs = run_for_one_layer(dataset, graphs, config, target_layer)

        dataset_precisions.append(np.array(precisions).mean())
        dataset_aucs.append(np.array(aucs).mean())

    print(f'{Fore.BLUE}\n--------------------------------------------')
    print('Dataset: ', dataset['name'])
    print('Dataset AVG Precision = ', round(np.array(dataset_precisions).mean(), 4))
    print('Dataset AVG AUC = ', round(np.array(dataset_aucs).mean(), 4))
    print(f'--------------------------------------------\n{Fore.RESET}')
    return dataset_precisions, dataset_aucs


def compare_intra_similarity_methods(dataset):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'{Style.BRIGHT}\tCompare Intra-layer Similarity Methods')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    config = Config()
    config.runs = 5
    config.intra_layer_similarity = 'PCC'
    config.target_layer = 0
    config.use_tuned_phi = False
    all_results = []
    graphs = import_dataset(path='./data/' + dataset["name"] + '/edges.csv',
                            number_of_nodes=dataset['number_of_nodes'], number_of_layers=dataset['layers'])
    for item in ['RA', 'Jaccard', 'CN', 'AA', 'PA']:
        results = []
        phis = []

        print('\nSimilarity Method: ', item)

        for i in range(10):
            config.intra_layer_similarity = item
            config.phi = i * 0.1
            phis.append(config.phi)
            precision, auc = run_one_iteration(i, dataset, graphs=graphs, config=config)
            results.append(precision)

        all_results.append(results)

    plot(np.array(phis),
         np.array(all_results).T,
         'phi',
         'AUC',
         name='compare_sim_methods_' + dataset['name'] + '_layer_' + str(config.target_layer),
         labels=['RA', 'Jaccard', 'CN', 'AA', 'PA']
         )
    print()


def print_dataset_name(dataset):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'{Style.BRIGHT}\t\t\tDataset: {dataset["name"]}')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def plot(x, y, x_label, y_label, name='', labels=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    y = y.round(2)
    ax.set_yticks(np.arange(y.min(), y.max() + (y.max() - y.min()) / 10, (y.max() - y.min()) / 10))

    if labels:
        for i, label in enumerate(labels):
            plt.plot(x, y[:, i], label=label, marker="o")
        plt.legend(loc="upper left")
    else:
        plt.plot(x, y, marker="o")
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.title(name)
    plt.grid()
    plt.savefig('./plots/' + name)


class Config:
    intra_layer_similarity = 'RA'
    inter_layer_similarity = random.choice(['PCC', 'AASN', 'OR'])
    use_tuned_phi = True
    runs = 10
    test_ratio = 0.1
    k_for_precision = 10
    phi = 0.7
    target_layer = 0


datasets = [
    {'number': 1, 'name': 'padgett', 'layers': 2, 'number_of_nodes': 16},
    {'number': 2, 'name': 'krackhardt', 'layers': 3, 'number_of_nodes': 21},
    {'number': 3, 'name': 'vicker', 'layers': 3, 'number_of_nodes': 29},
    {'number': 4, 'name': 'aarhus', 'layers': 5, 'number_of_nodes': 61},
    {'number': 5, 'name': 'lazega', 'layers': 3, 'number_of_nodes': 71},
    # {'number': 6, 'name': 'london_transport', 'layers': 3, 'number_of_nodes': 369},
    # {'number': 7, 'name': 'eu_airlines', 'layers': 37, 'number_of_nodes': 450},
    # {'number': 8, 'name': 'fao_trade', 'layers': 364, 'number_of_nodes': 214},
]

config = Config()
config.intra_layer_similarity = 'RA'
config.inter_layer_similarity = random.choice(['PCC', 'AASN', 'OR'])
config.use_tuned_phi = True

# number of iterations per layer
config.runs = 10
config.test_ratio = 0.1
config.k_for_precision = 10

all_precisions = []
all_aucs = []

compare_intra_similarity_methods(datasets[2])

for dataset in datasets:
    dataset_precision, dataset_auc = run_for_a_dataset(dataset, config)
    all_aucs.append(np.array(dataset_auc).mean())
    all_precisions.append(np.array(dataset_precision).mean())

print(f'{Fore.LIGHTYELLOW_EX}\n||||||||||||||||||||||||||||||||||||||||||||||')
print('Overall AVG Precision = ', round(np.array(all_precisions).mean(), 4))
print('Overall AVG AUC = ', round(np.array(all_aucs).mean(), 4))
print(f'||||||||||||||||||||||||||||||||||||||||||||||\n{Fore.RESET}')

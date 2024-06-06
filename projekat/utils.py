import scanpy as sc
import SpaGFT as spg
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(data_path, preprocess=False):
    """Reads data from a file and preprocesses it if preprocess is set to True.

    Args:
        data_path (string): Path to the data file.
        preprocess (bool, optional): Whether to preprocess the data. Defaults to False.

    Returns:
        AnnData: Anndata object containing the data.
    """
    
    adata = sc.read(data_path)
    
    if preprocess:
        sc.pp.filter_genes(adata, min_cells=10)
        # Normalization
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        
    # adding necessary columns for SpaGFT to work
    adata.obs['array_row'] = adata.obsm['spatial'][:, 0]
    adata.obs['array_col'] = adata.obsm['spatial'][:, 1]
        
    return adata

def perform_spagft(adata):
    """Performs SpaGFT on the data.

    Args:
        adata (AnnData): Anndata object containing the data.

    Returns:
        np.array: Array containing the results of SpaGFT, 1 if the gene is spatially variable, 0 otherwise.
    """
    
    n_genes = adata.n_vars

    (ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(adata, ratio_neighbors=1)
    
    gene_df = spg.detect_svg(adata,
                            spatial_info=['array_row', 'array_col'],
                            ratio_low_freq=ratio_low,
                            ratio_high_freq=ratio_high,
                            ratio_neighbors=1,
                            filter_peaks=True,
                            S=6)
    
    svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()

    ground_truth = np.zeros(n_genes)
    for i, gene in enumerate(adata.var_names):
        if gene in svg_list:
            ground_truth[i] = 1

    return ground_truth

def perform_confusion_analysis(ground_truth, predicted_svg, title):
    """Performs analysis on the results of SpaGFT.

    Args:
        ground_truth (np.array): Ground truth array.
        predicted_svg (np.array): Predicted array.

    Returns:
        none: Prints the accuracy, precision, recall, and F1 score.
    """
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_svg).ravel()

    conf_matrix = np.array([[tp, fn], [fp, tn]])

    # plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig('output/graphs/' + title + '.pdf')
    plt.show()

    # calculate and display statistics
    accuracy = accuracy_score(ground_truth, predicted_svg)
    precision = precision_score(ground_truth, predicted_svg)
    recall = recall_score(ground_truth, predicted_svg)
    f1 = f1_score(ground_truth, predicted_svg)

    print(title)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')

def perform_roc_analysis(ground_truth, predicted_svg, title):
    """Performs ROC analysis on the results of SpaGFT.

    Args:
        ground_truth (np.array): Ground truth array.
        predicted_svg (np.array): Predicted array.

    Returns:
        none: Plots the ROC curve and prints the AUC score.
    """
    
    fpr, tpr, _ = roc_curve(ground_truth, predicted_svg)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('output/graphs/' + title + '.pdf')
    plt.show()
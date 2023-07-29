
from preprocessor import Preprocessor
from generate_matrix import MatrixGenerator
from clustering import Cluster
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    data_preprocess = Preprocessor()
    matrix_generate = MatrixGenerator()
    cluster_process = Cluster()

    # process in files
    folder_paths = [
        'data/C1',
        'data/C4',
        'data/C7'
    ]

    # Get the file path
    txt_files = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(folder_path, file)
                txt_files.append(file_path)

    # Process the documents
    df_list = []
    tf_dict_list = []
    word_dict_list = []
    for file_path in txt_files:
        # Preprocess
        word_dict = data_preprocess.get_word_bags(file_path)
        no_stop_words = data_preprocess.stopwords_filter(file_path)

        # Document-Term matrix
        df = pd.DataFrame([word_dict])
        df_list.append(df)

        # Compute TF
        tf_dict = matrix_generate.compute_tf(word_dict, no_stop_words)
        tf_dict_list.append(tf_dict)
        word_dict_list.append(word_dict)

    print('========================Personal Information========================')
    print('Name: Kuan-Hsuan(Jennifer) Chiu')
    print('School: New York University')
    print('Major: Information Systems')
    print('Project: Text Mining')

    print('========================Document-Term Matrix========================')
    document_term_matrix = matrix_generate.generate_document_term_matrix(df_list, txt_files)
    print(document_term_matrix)

    print('========================TF-IDF Matrix========================')
    # Compute IDF
    idf_dict = matrix_generate.compute_idf(word_dict_list)

    # Compute TF-IDF
    tfidf_dict_list = []
    for tf in tf_dict_list:
        tfidf_dict = matrix_generate.compute_tfidf(tf, idf_dict)
        tfidf_dict_list.append(tfidf_dict)
    tfidf_matrix = matrix_generate.generate_tfidf_matrix(tfidf_dict_list, txt_files)
    print(tfidf_matrix)

    print('========================Keywords for Each Document Folder========================')
    # Seperate the 'tfidf_dict_list' into 3 sub_list containing 8 'tfidf_dict' entries
    sub_lists = []
    chunk_size = 8
    for i in range(0, len(tfidf_dict_list), chunk_size):
        sub_list = tfidf_dict_list[i:i + chunk_size]
        sub_lists.append(sub_list)

    # Generate keywords for each folder
    keywords_c1 = matrix_generate.generate_keywords(sub_lists, 0)
    keywords_c4 = matrix_generate.generate_keywords(sub_lists, 1)
    keywords_c7 = matrix_generate.generate_keywords(sub_lists, 2)

    print(f"C1:{keywords_c1}")
    print(f"C4:{keywords_c4}")
    print(f"C7:{keywords_c7}")

    # Create a file and write the keywords results to it
    with open('topics.txt', 'w') as file:
        file.write(f"Keywords for Each Document Folder:\n")
        file.write(f"C1:{keywords_c1}\n")
        file.write(f"C4:{keywords_c4}\n")
        file.write(f"C7:{keywords_c7}\n")

    print('========================K-Means========================')
    # PCA
    n_components = 2
    reduced_matrix = cluster_process.pca(tfidf_matrix, n_components)
    # print(reduced_matrix)

    # K-means
    k = 3
    centroids_eucli, labels_eucli = cluster_process.kmeans(reduced_matrix, k, distance_metric='euclidean')
    centroids_cosine, labels_cosine = cluster_process.kmeans(reduced_matrix, k, distance_metric='cosine')

    print(f"K-Means(Euclidean):{labels_eucli}")
    print(f"K-Means(Cosine):{labels_cosine}")

    print('========================Confusion Matrix========================')
    # Confusion Matrix
    true_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2])

    # Euclidean
    confusion_matrix_eucli, precision_eucli, recall_eucli, f1_score_eucli = cluster_process.confusion_matrix(true_labels, labels_eucli)
    print("------Euclidean------")
    print(confusion_matrix_eucli)
    print(f"Precision:{precision_eucli}")
    print(f"Recall:{recall_eucli}")
    print(f"F1-Score:{f1_score_eucli}")

    # Cosine
    confusion_matrix_cosine, precision_cosine, recall_cosine, f1_score_cosine = cluster_process.confusion_matrix(true_labels, labels_cosine)
    print("------Cosine------")
    print(confusion_matrix_cosine)
    print(f"Precision:{precision_cosine}")
    print(f"Recall:{recall_cosine}")
    print(f"F1-Score:{f1_score_cosine}")

    print('========================Visualization will be shown in pop-up windows========================')
    # Visualization
    # Original
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Samples')
    plt.show()

    # Cosine
    sns.scatterplot(x=reduced_matrix[:, 0], y=reduced_matrix[:, 1], hue=labels_cosine, palette='tab10')  # visualize cluster
    plt.scatter(centroids_cosine[:, 0], centroids_cosine[:, 1], marker='D', s=20, c='red', label='Centroids')  # visualize centroid
    for i, centroid in enumerate(centroids_cosine):
        plt.text(centroid[0], centroid[1], f'Centroid {i}', fontsize=6, ha='left', va='center', color='black')

    theme_labels = ['Airline Safety', 'Hoof Mouth Disease', 'Bank Mortgage']
    cluster_labels = [0, 1, 2]
    for i in range(len(theme_labels)):
        theme_start_idx = i * 8
        theme_end_idx = (i + 1) * 8
        theme_mid_idx = (theme_start_idx + theme_end_idx) // 2
        theme_x = reduced_matrix[theme_mid_idx, 0]
        theme_y = reduced_matrix[theme_mid_idx, 1]
        plt.text(theme_x, theme_y, theme_labels[i], fontsize=6, ha='right', va='center', color='blue')

    plt.title('K-Means(Cosine) clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(title='Cluster')
    plt.show()








if __name__ == '__main__':
    main()
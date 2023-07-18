
from preprocessor import Preprocessor
from generate_matrix import MatrixGenerator
from clustering import Cluster
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    print('========================PCA Reduced Matrix========================')
    # PCA
    n_components = 2
    reduced_matrix = cluster_process.pca(tfidf_matrix, n_components)
    print(reduced_matrix)

    print('========================K-means========================')
    # K-means
    k = 3
    labels, centroids = cluster_process.kmeans(reduced_matrix, k)
    print(f'labels:{labels}')
    print(f'centroids:{centroids}')


    # # plot
    # plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1])
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Scatter Plot of Samples')
    # plt.show()








if __name__ == '__main__':
    main()

from preprocessor import Preprocessor
import os


def main():
    data_preprocess = Preprocessor()

    # process in files
    folder_paths = [
        '/Users/Jennifer/Desktop/hw3/data/C1',
        '/Users/Jennifer/Desktop/hw3/data/C4',
        '/Users/Jennifer/Desktop/hw3/data/C7'
    ]

    # Get the path of all files
    txt_files = []
    for folder_path in folder_paths:
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(folder_path, file)
                txt_files.append(file_path)

    # Save word bags of each txt file into a list (no duplicate)
    all_word_bags = []
    for file_path in txt_files:
        each_word_bags = data_preprocess.get_word_bags(file_path)
        all_word_bags += each_word_bags
    final_word_bags = list(set(all_word_bags))

    # Print word bags list
    print(final_word_bags)








if __name__ == '__main__':
    main()
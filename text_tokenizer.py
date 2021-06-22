import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize


def text_tokenizer(input_directory, output_directory):
    cnt = 1

    for filename in os.listdir(input_directory):
        with open(input_directory + '/' + filename, "r") as i:
            data = i.read()

        with open(output_directory + '/' + filename + "_using_NLTK.txt", "w") as f:
            data = data.replace("\t", ' ')
            data = data.replace("\n", ' ')
            data = re.sub('e.g.\s', ' e.g.- ', data)

            sentences = sent_tokenize(data)
            for sentence in sentences:
                f.write("\n\n<----------------------------------- Sentence %s -------------------------------------------> \n" % cnt)
                f.write(sentence)
                words = word_tokenize(sentence)
                f.write("\n")
                for word in words:
                    f.write("\n--->>> %s" % word)
                cnt += 1


if __name__ == '__main__':
    print("\n\n\n *** Text Tokenization Program *** \n\n")
    input_file_directory = input("> Enter the input directory path:")
    output_file_directory = input("> Enter the output directory path:")
    text_tokenizer(input_file_directory, output_file_directory)

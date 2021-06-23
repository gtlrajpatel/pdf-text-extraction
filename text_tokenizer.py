import os

from nltk.tokenize import sent_tokenize, word_tokenize


def text_tokenizer(input_directory, output_directory):
    """
    This method tokenize the setences & words from the given text file
    """
    for filename in os.listdir(input_directory):
        with open(input_directory + '/' + filename, "r") as i:
            data = i.read()

        with open(output_directory + '/' + filename + "_using_NLTK.txt", "w") as f:
            data = data.replace("\n", ' ')
            data = data.replace("e.g.", 'e.g.-')
            data = data.replace("i.e.", 'i.e.-')

            sentences = sent_tokenize(data)
            cnt = 1
            for sentence in sentences:
                f.write("<----------------------------------- Sentence %s -------------------------------------------> \n" % cnt)
                f.write(sentence)
                words = word_tokenize(sentence)
                f.write("\n")
                cnt += 1
                for word in words:
                    f.write("--->>> %s \n" % word)


if __name__ == '__main__':
    print("\n\n\n *** Text Tokenization Program *** \n\n")
    input_file_directory = input("> Enter the input directory path:")
    output_file_directory = input("> Enter the output directory path:")
    text_tokenizer(input_file_directory, output_file_directory)

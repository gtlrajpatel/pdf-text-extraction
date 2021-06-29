import os
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = set(stopwords.words('english'))


def text_tokenizer(input_directory, output_directory):
    """
    This method tokenize the sentences & words from the given text file.
    After that it removes the stop words and assigns the appropriate PoS tag to each word.
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
                f.write("\n<----------------------------------- Sentence %s -------------------------------------------> \n" % cnt)
                f.write(sentence)
                words = word_tokenize(sentence)
                words = [word for word in words if not word in stop_words]
                tagged_words = nltk.pos_tag(words)
                f.write("\n\nPOS Tags are: \n>> %s \n" % tagged_words)
                f.write("\nTokens are: \n")
                cnt += 1
                for word in words:
                    f.write(">> %s \n" % word)


if __name__ == '__main__':
    print("\n\n\n *** Text Tokenization Program *** \n\n")
    input_file_directory = input("> Enter the input directory path:")
    output_file_directory = input("> Enter the output directory path:")
    text_tokenizer(input_file_directory, output_file_directory)

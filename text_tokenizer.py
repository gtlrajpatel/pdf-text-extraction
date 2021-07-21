import os
import nltk
import spacy
import stanza

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from nltk.chunk import ne_chunk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


def nltk_processing(input_directory, output_directory):
    """
    NLP Processing on non xml files using NLTK library
    """
    os.environ['STANFORD_PARSER'] = '/home/rajpatel/Downloads/stanford-parser-full-2020-11-17/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/rajpatel/Downloads/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'

    path_to_jar = '/home/rajpatel/Downloads/stanford-parser-full-2020-11-17/stanford-parser.jar'
    path_to_models_jar = '/home/rajpatel/Downloads/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    stop_words = set(stopwords.words('english'))

    for filename in os.listdir(input_directory):
        with open(input_directory + '/' + filename, "r") as i:
            data = i.read()

        filename, ext = os.path.splitext(os.path.basename(filename))
        with open(output_directory + '/' + filename + "_using_NLTK.txt", "w") as f:
            data = data.replace("\n", ' ')
            data = data.replace("e.g.", 'e.g.-')
            data = data.replace("eg.", 'e.g.-')
            data = data.replace("e.g", 'e.g.-')
            data = data.replace("i.e.", 'i.e.-')

            sentences = sent_tokenize(data)

            sentence_count = 0
            organization_count, person_count, gpe_count, bigram_count, trigram_count, \
                token_count_with_stopwords, token_count_without_stopwords, noun_phrase_count,\
                porter_words_count, snowball_words_count, lemmatize_words_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            f.write("\t\t\t\t *** Text Processing using NLTK *** \n")
            for sentence in sentences:
                f.write(f'\n\n============================ Sentence {sentence_count + 1} =============================')
                f.write("\n\n%s \n" % sentence)

                # tokenize the sentences into words
                words = word_tokenize(sentence)
                token_count_with_stopwords = token_count_with_stopwords + len(words)

                # checking for stop words
                words = [word for word in words if not word in stop_words]
                token_count_without_stopwords = token_count_without_stopwords + len(words)
                f.write("\n\n>> Tokens are: \n %s" % words)

                # extracting bigrams
                bigram = nltk.bigrams(words)
                bigrams = list(bigram)
                bigram_count += len(bigrams)
                f.write("\n\n>> Bigrams are: \n %s" % bigrams)

                # extracting trigrams
                trigram = nltk.trigrams(words)
                trigrams = list(trigram)
                trigram_count += len(trigrams)
                f.write("\n\n>> Trigrams are: \n %s" % trigrams)

                # labelling each word with appropriate PoS tag
                tagged_words = nltk.pos_tag(words)
                f.write("\n\n>> POS Tags are: \n %s" % tagged_words)

                # dependency parsing
                try:
                    result = dependency_parser.raw_parse(sentence)
                    dep = result.__next__()
                    f.write("\n\n>> Dependencies are: \n %s \n" % list(dep.triples()))
                except Exception as e:
                    print("\n\n\n Exception occur in file '%s'. \n\n Error is ---> %s" % (filename, e))

                # noun-phrase chunking
                pattern = 'NP: {<DT>?<JJ>*(<NN>|<NNP>|<NNS>)+}'
                cp = nltk.RegexpParser(pattern)
                cs = cp.parse(tagged_words).subtrees()
                f.write("\n\n %s \n" % cp.parse(tagged_words))

                noun_phrases_list = [' '.join(leaf[0] for leaf in tree.leaves()) for tree in cs if tree.label() == 'NP']
                f.write("\n\n>> Noun Phrases are: \n %s" % noun_phrases_list)

                # named entity recognition
                ne_tree = ne_chunk(tagged_words)
                ner_list = []
                for chunk in ne_tree:
                    if hasattr(chunk, 'label'):
                        ner_list.append((chunk.label(), ' '.join(c[0] for c in chunk)))

                f.write("\n\n>> Named Entities are: \n %s " % ner_list)
                for org in ner_list:
                    if org[0] == 'ORGANIZATION':
                        organization_count += 1
                    if org[0] == 'PERSON':
                        person_count += 1
                    if org[0] == 'GPE':
                        gpe_count += 1

                # stemming using porter stemmer
                porter_root_words = []
                porter_stemmer = PorterStemmer()
                for word in words:
                    root_word = porter_stemmer.stem(word)
                    porter_root_words.append((word, root_word))
                porter_words_count = porter_words_count + len(porter_root_words)
                f.write("\n\n>> Stemming using Porter Stemmer: \n %s" % porter_root_words)

                # stemming using snowball stemmer
                snowball_root_words = []
                snow_stemmer = SnowballStemmer(language='english')
                for word in words:
                    root_word = snow_stemmer.stem(word)
                    snowball_root_words.append((word, root_word))
                snowball_words_count = snowball_words_count + len(snowball_root_words)
                f.write("\n\n>> Stemming using Snowball Stemmer: \n %s" % snowball_root_words)

                # lemmatization
                wordnet_lemmatizer = WordNetLemmatizer()
                lemmatized_words = []
                for word in words:
                    lemmatized_word = wordnet_lemmatizer.lemmatize(word)
                    lemmatized_words.append((word, lemmatized_word))
                lemmatize_words_count = lemmatize_words_count + len(lemmatized_words)
                f.write("\n\n>> Lemmatization: \n %s\n\n" % lemmatized_words)

                sentence_count += 1

            f.write("\n\n\n\n \t\t\t\t ***************************** File Report ********************************* \n\n"
                    ">> Sentence Count: %s\n\n>> Tokens with stopwords: %s\n>> Tokens without stopwords: %s\n\n"
                    ">> Bigram Count: %s\n>> Trigram Count: %s\n\n"
                    ">> Organization Count: %s\n>> Person Count: %s\n>> GPE Count: %s\n\n"
                    ">> Porter Stemmer Words: %s\n>> Snowball Stemmer Words: %s\n>> Lemmatize Words: %s"
                    % (sentence_count, token_count_with_stopwords, token_count_without_stopwords,
                       bigram_count, trigram_count, organization_count, person_count, gpe_count,
                       porter_words_count, snowball_words_count, lemmatize_words_count))


def spacy_processing(input_directory, output_directory):
    """
    NLP Processing on non xml files using SpaCy library
    """
    nlp = spacy.load('en_core_web_sm')

    for filename in os.listdir(input_directory):
        with open(input_directory + '/' + filename, "r") as i:
            data = i.read()

        filename, ext = os.path.splitext(os.path.basename(filename))
        with open(output_directory + '/' + filename + "_using_Spacy.txt", "w") as f:
            data = data.replace("\n", ' ')
            data = data.replace("e.g.", 'e.g.-')
            data = data.replace("i.e.", 'i.e.-')

            doc = nlp(data)
            sentences = list(doc.sents)
            sentence_count, token_with_stopwords, token_without_stopwords, bigram_count, trigram_count,\
                noun_phrase_count, organization_count, person_count, gpe_count = 0, 0, 0, 0, 0, 0, 0, 0, 0

            f.write("\t\t\t\t *** Text Processing using Spacy *** \n")
            for sentence in sentences:
                f.write(f'\n\n============================ Sentence {sentence_count + 1} =============================')
                f.write("\n\n%s \n" % sentence)
                sentence_count += 1

                tokens, pos_tags, dep_tags = [], [], []
                for token in sentence:
                    token_with_stopwords = token_with_stopwords + 1
                    if not token.is_stop:
                        token_without_stopwords = token_without_stopwords + 1
                        tokens.append(token)
                        pos_tags.append((token.text, token.pos_))
                        dep_tags.append((token.text, token.dep_))

                f.write("\n\n>> Tokens are: \n%s \n\n>> PoS Tags are: \n%s \n\n>> Dependency Tags are: \n%s"
                        % (tokens, pos_tags, dep_tags))

                # bigrams generation
                bigrams = []
                for word in range(len(tokens)-1):
                    firstWord = tokens[word]
                    secondWord = tokens[word + 1]
                    element = [firstWord, secondWord]
                    bigrams.append(element)

                bigram_count += len(bigrams)
                f.write("\n\n>> Bigrams: \n%s" % bigrams)

                # trigrams generation
                trigrams = []
                for word in range(len(tokens)-2):
                    firstWord = tokens[word]
                    secondWord = tokens[word + 1]
                    thirdWord = tokens[word + 2]
                    element = [firstWord, secondWord, thirdWord]
                    trigrams.append(element)

                trigram_count += len(trigrams)
                f.write("\n\n>> Trigrams: \n%s" % trigrams)

                # noun-phrase chunking
                noun_phrase_count += len(list(sentence.noun_chunks))
                f.write("\n\n>> Noun Phrases are: \n%s" % list(sentence.noun_chunks))

                # named entity recognition
                f.write("\n\n>> Named Entities are: \n%s\n" % [(ent.text, ent.label_) for ent in sentence.ents])

                for ent in sentence.ents:
                    if ent.label_ == 'ORG':
                        organization_count += 1
                    if ent.label_ == 'PERSON':
                        person_count += 1
                    if ent.label_ == 'GPE':
                        gpe_count += 1

            f.write("\n\n\n\n \t\t\t\t ***************************** File Report ********************************* \n\n"
                    ">> Sentence Count: %s\n\n>> Token with stopwords: %s\n>> Token without stopwords: %s\n\n"
                    ">> Total Bigrams: %s\n>> Total Trigrams: %s\n\n>> Total Noun Phrase: %s\n\n"
                    ">> Organization Count: %s\n>> Person Count: %s\n>> GPE Count: %s"
                    % (sentence_count, token_with_stopwords, token_without_stopwords, bigram_count, trigram_count,
                       noun_phrase_count, organization_count, person_count, gpe_count))


def stanza_processing(input_directory, output_directory):
    """
    NLP Processing on non xml files using Stanza library
    """
    nlp = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, ner, depparse')
    for filename in os.listdir(input_directory):
        with open(input_directory + '/' + filename, "r") as i:
            data = i.read()

        filename, ext = os.path.splitext(os.path.basename(filename))
        with open(output_directory + '/' + filename + "_using_Stanza.txt", "w") as f:
            data = data.replace("\n", ' ')
            data = data.replace("e.g.", 'e.g.-')
            data = data.replace("i.e.", 'i.e.-')

            sentence_count = 0
            token_count_with_stopwords, lemma_count, organization_count, person_count, gpe_count = 0, 0, 0, 0, 0
            doc = nlp(data)
            f.write("\t\t\t\t *** Text Processing using Stanza *** \n")
            for i, sentence in enumerate(doc.sentences):
                f.write(f'\n\n========================== Sentence {i + 1} ===========================')
                f.write("\n\n %s \n" % sentence.text)
                f.write("\nTokens are: ")
                tokens = []
                for token in sentence.tokens:
                    tokens.append(token.text)
                token_count_with_stopwords += len(tokens)
                f.write("\n>> %s " % tokens)

                upos, xpos, lemma, dep_parse = [], [], [], []
                for word in sentence.words:
                    upos.append((word.text, word.upos))
                    xpos.append((word.text, word.xpos))
                    lemma.append((word.text, word.lemma))

                    head = sentence.words[word.head - 1].text if word.head > 0 else 'root'
                    dep_parse.append(((word.text, head), word.deprel))

                lemma_count += len(lemma)
                f.write("\n\n UPOS tags are: \n>> %s \n\n XPOS tags are: \n>> %s \n\n Lemmas are: \n>> %s "
                        "\n\n Dependency tags are: \n>> %s" % (upos, xpos, lemma, dep_parse))

                ner = []
                for ent in sentence.ents:
                    if ent.type == 'ORG':
                        organization_count += 1
                    if ent.type == 'GPE':
                        gpe_count += 1
                    if ent.type == 'PERSON':
                        person_count += 1
                    ner.append((ent.text, ent.type))
                f.write("\n\n Named Entities are: \n>> %s" % ner)
                sentence_count += 1

            f.write("\n\n\n\n \t\t\t\t ***************************** File Report ********************************* \n\n"
                    ">> Sentence Count: %s\n\n>> Total Tokens: %s\n\n>> Total Lemmas: %s\n\n"
                    ">> Organization Count: %s\n>> Person Count: %s\n>> GPE Count: %s"
                    % (i + 1, token_count_with_stopwords, lemma_count, organization_count, person_count, gpe_count))


if __name__ == '__main__':
    print("\n\n\n *** NLP Processing on non xml files *** \n\n")
    print("Please select the processing method:\n"
          "1. NLP processing using NLTK\n"
          "2. NLP processing using Spacy\n"
          "3. NLP processing using Stanza\n")
    user_choice = input("\nPlease Enter your choice (1, 2 or 3):")

    input_file_directory = input("\n> Enter the input directory path:")
    output_file_directory = input("\n> Enter the output directory path:")

    if user_choice == '1':
        print("\n\n NLTK in Progress...")
        nltk_processing(input_file_directory, output_file_directory)
    elif user_choice == '2':
        print("\n\n Spacy in Progress...")
        spacy_processing(input_file_directory, output_file_directory)
    elif user_choice == '3':
        print("\n\n Stanza in Progress...")
        stanza_processing(input_file_directory, output_file_directory)
    else:
        print("Invalid Choice")

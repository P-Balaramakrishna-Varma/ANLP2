import nltk
import string



# read the entrire file into a string
def get_tokens_from_text_corpus(text):
    # word tokenizer.
    tokens = nltk.word_tokenize(text)

    # lower casing
    tokens = [word.lower() for word in tokens]

    # removing puncuations
    punctuations = string.punctuation + "\u201C" + "\u201D" + "\u2019" + "\u2018"
    tokens = [word for word in tokens if word not in punctuations]

    # handling words which have a period at the end.
    tokens = [word[:-1] if word[-1] == '.' else word for word in tokens]
    return tokens



if __name__ == '__main__':
    while True:
        input_text = input("Enter the text to tokenize: ")
        tokens = get_tokens_from_text_corpus(input_text)
        print(tokens)
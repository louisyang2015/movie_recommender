"""
This module searches the list of titles using idf.
Run the program on the command line, which builds an Index
object, and calls print_xxx(...) functions to show current
status of the index. Improve "IndexerConfig" as necessary.

Command line flags:
    overwrite - sets the overwrite flag, overwriting existing .bin files
        default is False
"""
import os, pickle, sys, textwrap
import movie_lens_data
from PorterStemmer import PorterStemmer



class IndexerConfig:
    """ Configuration for the Indexer object
    ::
        ignored_words - set of ignored words - see define_ignored_words_set()
        porter_stemmer - PorterStemmer instance
        word_stems - {word: word stem} - see define_word_stems()
    """

    def __init__(self):
        self.ignored_words = self.define_ignored_words_set()
        self.porter_stemmer = PorterStemmer()
        self.word_stems = self.define_word_stems()


    def define_ignored_words_set(self):
        """Return a set of words to be ignored."""
        # set of words to ignore during tokenization
        ignored_words = set()

        # add (1900) through (2300)
        for i in range(1900, 2301):
            ignored_words.add("(" + str(i) + ")")

        ignored_words.update(["the", "of", "a", "and", "in",
                              "to", "on", "&", "for", "la",
                              "a.k.a"])

        return ignored_words


    def define_word_stems(self):
        """Return a {word: stem word} for this data set."""
        return {
            "m*a*s*h": "mash",
            "u.s.a": "usa",
            "m.d": "md",
            "l.a": "la",
            "d.a": "da",
            "s.w.a.t": "swat",
            "r.i.p.d": "ripd",
            "i.q": "iq",
            "a.i": "ai",
            "s.h.i.e.l.d": "shield",
            "a.d": "ad",
            "kissing": "kiss",
            "ii": "2",
            "iii": "3",
            "iv": "4",
            "v": "5",
            "vi": "6",
            "vii": "7",
            "viii": "8",
            "ix": "9",
            "x": "10",
            "u.f.o": "ufo",
            "s*p*y*": "spy",
            "9/11": "911"
        }


    def text_splitter(self, text : str):
        """Break text into a list of words."""
        words = text.split()

        # additional splits
        words2 = []

        for word in words:
            # split on '-'
            if '-' in word:
                tokens = word.split('-')

                for token in tokens:
                    if len(token) > 0:
                        words2.append(token)

            # split on "/"
            elif (word.count("/") == 1) and (len(word) >= 6):
                tokens = word.split('/')
                words2.append(tokens[0])
                words2.append(tokens[1])

            else:
                words2.append(word)

        return words2


    def word_filter(self, word : str):
        word = word.lower()

        # check for ignored words
        if word in self.ignored_words:
            return None

        old_length = len(word)

        word = self.remove_punctuation(word)

        word = self.stem(word)

        if len(word) < 1: return None

        # check for ignored words a second time as needed
        if len(word) < old_length:
            if word in self.ignored_words:
                return None

        return word


    def remove_punctuation(self, word : str):
        """Return the same word, with punctuation marks removed."""

        start_punctuation = ['(', '*', '.', '\'']
        end_punctuation = [')', ',', ':', '?!', '!?',
                           '!', '?', '.', '+', '\'',
                           '*']

        for p in start_punctuation:
            if word.startswith(p):
                word = word.lstrip(p)

        for p in end_punctuation:
            if word.endswith(p):
                word = word.rstrip(p)

        return word


    def stem(self, word : str):
        """Return a word's base (stem) form."""
        # remove ending "'s"
        if word.endswith("'s"):
            word = word[:-2]

        # check with stem dictionary
        if word in self.word_stems:
            word = self.word_stems[word]

        return self.porter_stemmer.stem(word, 0, len(word) - 1)




class Index:
    """ Class for indexing movie titles.
    ::
        config - an IndexerConfig object
        tokens_index - {word: set of movie ids}
        tokens_count - {word: count}
        bigrams_index - {word: set of movie ids}
        bigrams_count - {bigram: count}
    """

    def __init__(self, config : IndexerConfig, file_name):
        """Builds the index.
        ::
            config - an IndexerConfig object
            file_name - if None, then the index built is in complete. Call
                "build(...)" to complete the index.
        """
        self.config = config
        self.tokens_index, self.tokens_count = None, None
        self.bigrams_index, self.bigrams_count = None, None

        if file_name is not None:
            with open(file_name, mode="rb") as file:
                self.tokens_index, self.tokens_count, self.bigrams_index, \
                    self.bigrams_count = pickle.load(file)



    def build(self, movie_titles : dict, file_name = None):
        """Builds the index, filling out "tokens_index", "tokens_count",
        "bigrams_index", and "bigrams_count".
        :param movie_titles: {movie_id: title}
        :param file_name: saves the index to this file name
        """
        # collect tokens and bigrams
        # tokens - {movie_id: list of words}
        # bigrams - {movie_id: list of bigrams (two words)}
        tokens, bigrams = self.tokenize_titles(movie_titles)

        # build indices and counts
        self.tokens_index, self.bigrams_index = self.build_reverse_indices(tokens, bigrams)
        self.tokens_count, self.bigrams_count = self.compute_word_counts(tokens, bigrams)

        print("The index has", len(tokens), "titles.")

        if file_name is not None:
            with open(file_name, mode="wb") as file:
                obj = [self.tokens_index, self.tokens_count, self.bigrams_index,
                       self.bigrams_count]
                pickle.dump(obj, file)


    def tokenize_titles(self, movie_titles):
        """Process movie titles. Returns tokens, bigrams.
        ::
            Inputs
                movie_titles - {movie_id: title}

            Returns
                tokens - {movie_id: list of words}
                bigrams - {movie_id: list of bigrams (two words)}
        """
        tokens = {} # {movie_id: list of words}
        bigrams = {}  # {movie_id: list of bigrams (two words)}

        for movie_id in movie_titles:
            title = movie_titles[movie_id]

            tokens_list, bigrams_list = self.tokenize(title)

            if tokens_list is not None:
                tokens[movie_id] = tokens_list

            if bigrams_list is not None:
                bigrams[movie_id] = bigrams_list

        return tokens, bigrams


    def tokenize(self, text : str):
        """Returns tokens and bigrams. Both are list of words."""
        # extract "tokens" from text
        tokens = []

        words = self.config.text_splitter(text)

        for word in words:
            word = self.config.word_filter(word)

            if word is not None:
                tokens.append(word)

        # extract "bigrams" from tokens
        bigrams = []
        if len(tokens) >= 2:
            for i in range(0, len(tokens) - 1):
                bigrams.append(tokens[i] + " " + tokens[i+1])

        if len(tokens) < 1: tokens = None
        if len(bigrams) < 1: bigrams = None

        return tokens, bigrams


    def build_reverse_indices(self, tokens, bigrams):
        """Creates indices that maps from terms to movie_id.
        ::
            Inputs
                tokens - {movie_id: list of words}
                bigrams - {movie_id: list of bigrams (two words)}

            Returns
                tokens_index - {word: set of movie ids}
                bigrams_index - {word: set of movie ids}
        """
        tokens_index = {}

        for movie_id in tokens:
            for word in tokens[movie_id]:

                if word not in tokens_index:
                    tokens_index[word] = set()

                tokens_index[word].add(movie_id)

        bigrams_index = {}

        for movie_id in bigrams:
            for bigram in bigrams[movie_id]:

                if bigram not in bigrams_index:
                    bigrams_index[bigram] = set()

                bigrams_index[bigram].add(movie_id)

        return tokens_index, bigrams_index


    def compute_word_counts(self, tokens, bigrams):
        """
        ::
            Inputs
                tokens - {movie_id: list of words}
                bigrams - {movie_id: list of bigrams (two words)}

            Returns
                tokens_count - {word: count}
                bigrams_count - {bigram: count}
        """
        tokens_count = {}

        for movie_id in tokens:
            for word in tokens[movie_id]:
                if word not in tokens_count:
                    tokens_count[word] = 0

                tokens_count[word] += 1

        bigrams_count = {}

        for movie_id in bigrams:
            for bigram in bigrams[movie_id]:
                if bigram not in bigrams_count:
                    bigrams_count[bigram] = 0

                bigrams_count[bigram] += 1

        return tokens_count, bigrams_count


    def print_frequent_tokens(self, num_result : int):
        print("Frequent tokens in the index:")
        self.print_frequent_items(self.tokens_count, num_result)
        print()


    def print_frequent_bigrams(self, num_result : int):
        print("Frequent bigrams in the index:")
        self.print_frequent_items(self.bigrams_count, num_result)
        print()


    def print_frequent_items(self, items_dict : dict, num_result : int):
        """
        :param items_dict: {string: count}
        :param num_result: print only the most common results
        """
        items_list = list(items_dict.items())
        items_list.sort(key = lambda e: e[1], reverse = True)
        items_list = items_list[:num_result]

        print("count".center(15))
        for value, count in items_list:
            print(str(count).center(15), value)


    def print_non_alpha_num_words(self):
        """Print words that are not alpha numeric."""
        non_alpha_words = set()

        for token in self.tokens_count.keys():
            if token.isalnum() == False:
                non_alpha_words.add(token)

        print(len(non_alpha_words), "non-alphabetical words in index:")
        print(textwrap.fill(str(non_alpha_words)))
        print()


    def print_words_with_ending(self, ending : str):
        words = set()

        for token in self.tokens_count.keys():
            if token.endswith(ending):
                words.add(token)

        print(len(words), "words ending in \"" + ending + "\":")
        print(textwrap.fill(str(words)))
        print()


    def search(self, text : str):
        """Search the titles. Returns a list of
        (score, movie id) pairs."""

        tokens, bigrams = self.tokenize(text)

        tokens_index = self.tokens_index
        tokens_count = self.tokens_count
        bigrams_index = self.bigrams_index
        bigrams_count = self.bigrams_count

        scores = {} # {movie_id: score}

        # bigram matches are worth more than corresponding token matches
        # For example, matching "dragon ball" should be better than
        # matching "dragon" and "ball" separately
        bigram_multiplier = 1.5

        # accumulate score
        for word in tokens:
            if word in tokens_index:
                token_count = tokens_count[word]

                for movie_id in tokens_index[word]:

                    if movie_id not in scores:
                        scores[movie_id] = 0

                    scores[movie_id] += 1.0 / token_count

        if bigrams is not None:
            for bigram in bigrams:
                if bigram in bigrams_index:
                    bigram_count = bigrams_count[bigram]

                    for movie_id in bigrams_index[bigram]:

                        if movie_id not in scores:
                            scores[movie_id] = 0

                        scores[movie_id] += bigram_multiplier / bigram_count

        # sort score
        score_list = []
        for movie_id in scores.keys():
            score_list.append((scores[movie_id], movie_id))

        score_list.sort(reverse=True)
        return score_list




def search(index : Index, text : str, movie_titles : dict):
    """Searches the "index" for "text".
    ::
        movie_titles - {movie_id: movie titles}
    """
    print("Searching for:", text)

    results = index.search(text)
    results = results[:100]

    print("score".center(25), "movie ID".center(15), "Title".center(20))

    for result in results:
        movie_id = result[1]

        print("{0:.4g}".format(result[0]).center(25), # score
              str(movie_id).center(15),
              movie_titles[movie_id])

    print()




def main():
    # redirect output to file to avoid Unicode printing errors
    print('see output in "title_search_output.txt"')
    sys.stdout = open("title_search_output.txt", "w", encoding="utf-8")

    # process command line arguments
    overwrite = False

    for arg in sys.argv:
        if arg == "overwrite":
            overwrite = True

    # movie_titles is {movie_id: title}
    movie_lens_data.read_movies_csv(overwrite)
    movie_titles = movie_lens_data.get_input_obj("movie_titles")

    # get "index", either load it from disk, or rebuild it
    index = None
    index_file_name = movie_lens_data.out_dir + "title_search_index.bin"

    if os.path.exists(index_file_name) and overwrite == False:
        config = IndexerConfig()
        index = Index(config, index_file_name)
    else:
        # build an "Index" object using "movie_titles"
        config = IndexerConfig()
        index = Index(config, None)
        index.build(movie_titles, index_file_name)

    # print properties of the index
    index.print_frequent_tokens(100)
    index.print_frequent_bigrams(50)
    index.print_non_alpha_num_words()

    index.print_words_with_ending("s")
    index.print_words_with_ending("ing")
    index.print_words_with_ending("ion")

    # print some searches with the index
    search(index, "star war", movie_titles) # should match "star wars"
    search(index, "star trek 2", movie_titles) # should match "star trek ii"
    search(index, "battle of gods dragon ball", movie_titles) # prioritize "dragon ball" movies
    search(index, "ad", movie_titles) # should match "a.d"
    search(index, "shield", movie_titles) # should match "S.H.I.E.L.D."



if __name__ == "__main__":
    main()


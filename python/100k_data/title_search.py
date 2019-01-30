"""
This module searches the list of titles using idf.
"""
import csv, textwrap, os
from PorterStemmer import PorterStemmer


data_dir = "data" + os.sep


class CsvData:
    """This class contains data read from csv files.
    ::
        movie_titles - {movie id: title string}
    """

    def __init__(self):
        _, _, self.movie_titles = self.read_movies_csv()


    def read_movies_csv(self):
        """ Read "movies.csv". Returns genre_str_to_id,
        movie_id_to_genre_ids, movie_id_to_title.
        ::
            genre_str_to_id - {genre string:  genre id}
            movie_id_to_genre_ids - {movie id: set of genre ids}
            movie_id_to_title - {movie id: title string}
        """
        genre_str_to_id = {} # {genre string: genre id}
        next_genre_id = 0

        movie_id_to_genre_ids = {} # {movie id: set of genre ids}

        movie_id_to_title = {} # # {movie_id: title string}

        # read from "movies.csv" to collect genres
        with open(data_dir + "movies.csv", "r", encoding="utf-8", newline='') as file:
            csv_reader = csv.reader(file, delimiter=',')
            next(csv_reader)  # skip the first row

            for row in csv_reader:
                movie_id = int(row[0])
                title = row[1]
                genre_strings = row[2].lower().split('|')

                movie_id_to_title[movie_id] = title

                # go through each genre found
                for genre_str in genre_strings:
                    if genre_str != "(no genres listed)":

                        # add "genre_str" to "genre_str_to_id"
                        if genre_str not in genre_str_to_id:
                            genre_str_to_id[genre_str] = next_genre_id
                            next_genre_id += 1

                        genre_id = genre_str_to_id[genre_str]

                        # add genre_id to "movie_id_to_genre_ids"
                        if movie_id not in movie_id_to_genre_ids:
                            movie_id_to_genre_ids[movie_id] = set()

                        movie_id_to_genre_ids[movie_id].add(genre_id)

        return genre_str_to_id, movie_id_to_genre_ids, movie_id_to_title



class IndexerConfig:
    """ Configuration for the Indexer object
    ::
        ignored_words - set of ignored words
        porter_stemmer - PorterStemmer instance
        word_stems - {word: word stem}
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
            "kissing": "kiss"
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
        config
        csv_data

        bigrams - {movie_id: list of bigrams (two words)}
        tokens - {movie_id: list of words}
        tokens_index - {word: set of movie ids}
        bigrams_index - {word: set of movie ids}
        tokens_count - {word: count}
        bigrams_count - {bigram: count}
    """

    def __init__(self, csv_data : CsvData, config : IndexerConfig):
        self.csv_data = csv_data
        self.config = config
        self.tokens, self.bigrams = self.tokenize_titles()
        self.tokens_index, self.bigrams_index = self.build_reverse_indices()
        self.tokens_count, self.bigrams_count = self.compute_word_counts()

        print("The index has", len(self.tokens), "titles.")


    def tokenize_titles(self):
        """Process movie titles. Returns tokens, bigrams.
        ::
            tokens - {movie_id: list of words}
            bigrams - {movie_id: list of bigrams (two words)}
        """
        movie_titles = self.csv_data.movie_titles

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


    def build_reverse_indices(self):
        """Creates indices that maps from terms to movie_id.
        Returns tokens_index, bigrams_index.
        ::
            tokens_index - {word: set of movie ids}
            bigrams_index - {word: set of movie ids}
        """
        tokens = self.tokens
        tokens_index = {}

        for movie_id in tokens:
            for word in tokens[movie_id]:

                if word not in tokens_index:
                    tokens_index[word] = set()

                tokens_index[word].add(movie_id)

        bigrams = self.bigrams
        bigrams_index = {}

        for movie_id in bigrams:
            for bigram in bigrams[movie_id]:

                if bigram not in bigrams_index:
                    bigrams_index[bigram] = set()

                bigrams_index[bigram].add(movie_id)

        return tokens_index, bigrams_index


    def compute_word_counts(self):
        """Returns tokens_count, bigrams_count.
        ::
            tokens_count - {word: count}
            bigrams_count - {bigram: count}
        """
        tokens = self.tokens
        tokens_count = {}

        for movie_id in tokens:
            for word in tokens[movie_id]:
                if word not in tokens_count:
                    tokens_count[word] = 0

                tokens_count[word] += 1

        bigrams = self.bigrams
        bigrams_count = {}

        for movie_id in bigrams:
            for bigram in bigrams[movie_id]:
                if bigram not in bigrams_count:
                    bigrams_count[bigram] = 0

                bigrams_count[bigram] += 1

        return tokens_count, bigrams_count


    def print_frequent_tokens(self, num_result : int):
        print("Frequent tokens in the index:")
        self.print_frequent_words(self.tokens, num_result)
        print()


    def print_frequent_bigrams(self, num_result : int):
        print("Frequent bigrams in the index:")
        self.print_frequent_words(self.bigrams, num_result)
        print()


    def print_frequent_words(self, title_words, num_result : int):
        """
        :param title_words: {movie_id: list of words}
        :param num_result: number of most frequent words to print
        """

        # count words
        word_count = {} # {word: count}

        for movie_id in title_words:
            words = title_words[movie_id]

            for word in words:
                if word not in word_count:
                    word_count[word] = 0

                word_count[word] += 1

        # print top 100 words
        word_count_list = []
        for word in word_count:
            word_count_list.append((word_count[word], word))

        word_count_list.sort(reverse=True)
        word_count_list = word_count_list[:num_result]

        print("count".center(15), "word")
        for count, word in word_count_list:
            print(str(count).center(15), word)


    def print_non_alpha_num_words(self):
        """Print words that are not alpha numeric."""
        non_alpha_words = set()

        for movie_id in self.tokens:
            tokens = self.tokens[movie_id] # list of words

            for token in tokens:
                if token.isalnum() == False:
                    non_alpha_words.add(token)

        print(len(non_alpha_words), "non-alphabetical words in index:")
        print(textwrap.fill(str(non_alpha_words)))
        print()


    def print_words_with_ending(self, ending : str):
        words = set()

        for movie_id in self.tokens:
            tokens = self.tokens[movie_id] # list of words

            for token in tokens:
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



def search(index : Index, text : str):
    """Searches the "index" for "text"."""
    print("Searching for:", text)

    results = index.search(text)
    results = results[:100]

    title_lookup = index.csv_data.movie_titles

    print("score".center(25), "movie ID".center(15), "Title".center(20))

    for result in results:
        movie_id = result[1]

        print("{0:.4g}".format(result[0]).center(25), # score
              str(movie_id).center(15),
              title_lookup[movie_id])

    print()




def main():
    csv_data = CsvData()
    config = IndexerConfig()
    index = Index(csv_data, config)

    index.print_frequent_tokens(100)
    index.print_frequent_bigrams(50)
    index.print_non_alpha_num_words()

    index.print_words_with_ending("s")
    index.print_words_with_ending("ing")
    index.print_words_with_ending("ion")

    search(index, "star war")
    search(index, "star trek")
    search(index, "battle of gods dragon ball")
    search(index, "ad")
    search(index, "shield")




if __name__ == "__main__":
    main()


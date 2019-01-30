"""A description of project's files"""

# Files that will be copied from their upstream locations as
# needed.
upstream_files = {
    "genre_counts.bin": "../full_data/data/in/genre_counts.bin",
    "genre_ids.bin": "../full_data/data/in/genre_ids.bin",
    "movie_genres.bin": "../full_data/data/in/movie_genres.bin",
    "movie_tags.bin": "../full_data/data/in/movie_tags.bin",
    "tag_counts.bin": "../full_data/data/in/tag_counts.bin",
    "tag_ids.bin": "../full_data/data/in/tag_ids.bin",
    
    "movie_titles.bin": "../full_data/data/out/movie_titles.bin",
    "similar_movies.bin": "../full_data/data/out/similar_movies.bin",
    "title_search_index.bin": "../full_data/data/out/title_search_index.bin",
    "tmdb_data.bin": "../full_data/data/out/tmdb_data.bin"
}

# Files shared in different APIs that will be updated to the
# latest version.
shared_files = {"movie_data.py", "user_data.py"}

# Lambda API files, one directory per API.
lambda_projects = {
    "get_rated_movies": {"get_rated_movies.py", "movie_data.py",
                         "movie_titles.bin", "tmdb_data.bin", "user_data.py"},
    "rate": {"rate.py", "user_data.py"},
    "recommend": {"als3_item_factors.bin", "als3_movie_ids.bin",
                  "als5_item_factors.bin", "als5_movie_ids.bin",
                  "als7_item_factors.bin", "als7_movie_ids.bin",
                  "als9_item_factors.bin", "als9_movie_ids.bin",
                  "als11_item_factors.bin", "als11_movie_ids.bin",
                  "genre_counts.bin", "genre_ids.bin", "models.py", 
                  "movie_data.py", "movie_genres.bin", 
                  "movie_medians_full.bin", "movie_tags.bin", 
                  "movie_titles.bin", "recommend.py", "tag_counts.bin", 
                  "tag_ids.bin", "tmdb_data.bin", "user_data.py"},
    "search": {"movie_data.py", "movie_titles.bin", "PorterStemmer.py",
               "search.py", "title_search_index.bin", "title_search_index.py",
               "tmdb_data.bin", "user_data.py"},
    "similar": {"movie_data.py", "movie_titles.bin", "similar.py",
                "similar_movies.bin", "tmdb_data.bin", "user_data.py"}
}

###########################################################
# S3 settings:
s3_bucket_name = "xxxxxxxxxxxx"
# Upstream and shared files are uploaded to S3 just once, at:
s3_shared_directory = "_shared"



"""
Before using this script:
    api_key needs to be set
output files: 
    "tmdb_data.bin" - {movie_id: [tmdb_id_str, poster_file_name]}
    "tmdb_bad_id.bin" - {movie_id: tmdb_id_str}
"""

import csv, json, os, pickle, requests, time
import config

api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # TMDB API key

pause_time = 2 # 2 second pause between TMDB API calls

links_file_name = r".\data\txt\links.csv"
tmdb_data_file_name = config.out_dir + "tmdb_data.bin"
tmdb_bad_id_file_name = config.out_dir + "tmdb_bad_id.bin"


def download_movie_data_from_tmdb(tmdb_id):
    """
    :param tmdb_id: string
    :return: poster_file_name, which can be "None" if the movie
        has no poster. Return False if there is an error.
    """
    try:
        # URL = https://api.themoviedb.org/3/movie/284053?api_key=xxxx
        url = "https://api.themoviedb.org/3/movie/" + tmdb_id + "?api_key=" + api_key

        response = requests.request("GET", url)
        data = json.loads(response.text)
        time.sleep(pause_time)

        return data["poster_path"]

    except Exception as ex:
        # some tmdb_ids are known to be wrong
        print("Exception:", ex)
        return False


def save_to_disk(tmdb_data, tmdb_bad_id):
    """Save "tmdb_data" and "tmdb_bad_id" to disk."""
    print("Saving data to disk")
    with open(tmdb_data_file_name, mode="wb") as file:
        pickle.dump(tmdb_data, file)

    with open(tmdb_bad_id_file_name, mode="wb") as file:
        pickle.dump(tmdb_bad_id, file)


def main():
    tmdb_data = {} # {movie_id: [tmdb_id_str, poster_file_name]}
    tmdb_bad_id = {} # to record entries with tmdb ID, but failed API call

    # initialize "tmdb_data" and "tmdb_bad_id" from if found on disk
    if os.path.exists(tmdb_data_file_name):
        with open(tmdb_data_file_name, mode="rb") as file:
            tmdb_data = pickle.load(file)

    if os.path.exists(tmdb_bad_id_file_name):
        with open(tmdb_bad_id_file_name, mode="rb") as file:
            tmdb_bad_id = pickle.load(file)

    print(len(tmdb_data), "movie data have been downloaded.")
    print(len(tmdb_bad_id), "bad TMDB ID encountered.")

    last_save_time = time.time()

    # go through the CSV file
    with open(links_file_name, "r", encoding="utf-8", newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip the first row

        for row in csv_reader:
            # extract movie_id and tmdb_id
            movie_id = int(row[0])

            tmdb_id_str = None
            if len(row) >= 3 and len(row[2]) > 0:
                tmdb_id_str = row[2]

            # download movie data from TMDB
            if (tmdb_id_str is not None) and (movie_id not in tmdb_data)\
                    and (movie_id not in tmdb_bad_id):
                print("Downloading data for movie ID", movie_id,
                      "TMDB ID", tmdb_id_str)
                poster_file_name = download_movie_data_from_tmdb(tmdb_id_str)

                # update "tmdb_data" and "tmdb_no_data"
                if poster_file_name == False:
                    tmdb_bad_id[movie_id] = tmdb_id_str
                else:
                    tmdb_data[movie_id] = [tmdb_id_str, poster_file_name]

            # save to disk periodically
            if time.time() - last_save_time > 60:
                save_to_disk(tmdb_data, tmdb_bad_id)
                last_save_time = time.time()

    # one last save
    save_to_disk(tmdb_data, tmdb_bad_id)
    print("All data has been downloaded.")


if __name__ == "__main__":
    main()


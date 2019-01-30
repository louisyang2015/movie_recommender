# Movie Recommender
Movie Recommender using the MovieLens dataset.

## Features
* linear and ALS models
* similar movies
* uses the full dataset

## [Video Intro (2 minutes)](https://youtu.be/4ZKTAM-SEpY)

## Running the App in Local Mode
The movie recommender app is designed for AWS, but can also run locally. The code assumes that port 8000 is available.

**Requirements:**
* Python 3
* NumPy, SciPy

Change to the "app_local" directory. For example:

    cd E:\proj2018\movies_recommend\python\app_local

**Run the Python server:**

    python server.py
    
*Under Windows, try to avoid clicking inside the "python server.py" command prompt Window. Doing so seems to freeze the Python interpreter until a key is pressed. If an API request is made during this "freeze" time period, there will be no response. Even after a key is pressed, the software might crash. In the event of software crash, rerun "python server.py".*

**Open the website in a browser** - double click "python/app_local/web_page/index.html"


# Running the Full Project

## Software Requirements
* Python 3, with NumPy, SciPy, boto3, requests
* C++ compiler

## Preparations
**Installing the MovieLens dataset**

* This project uses the following file: links.csv, movies.csv, ratings.csv, tags.csv
* Copy the small dataset files to "python/100k_data/data/"
* Copy the large dataset files to "python/full_data/data/txt/"

**The following network port numbers are assumed to be available:**

* 8000 - used by "python/app_local/server.py". Change via the "port_number" variable.
* 10000 - used by "cluster_server.py" files. Change via the "port_number" variable.
* 10001 - used by "worker_server.py" files. Change via the "port_number" variable.

## Necessary Code Changes

You do not have to do the following right away. You can return to this list when the code does not work.

**TMDB API Key**

For the "download_tmdb_data.py" file to work, the "api_key" variable needs to be specified.

**AWS Lambda Build Bucket**

For the "python/app/build.py" and "ec2_build.py" files to work, the "project_files.py" needs to have a valid "s3_bucket_name" variable.

**AWS Lambda API Endpoints**

The API endpoints at "python/app/web_page/js/global.js", under the "lambda_apis" variable, need to be updated.


## Documentation
Under /doc:
* **presentation.pptx** - instructions for running the full project. There is a video presentation:
    * [Part 1 - Recommendation Algorithms, slides #1 ~ 10 (1 hour, 22 minutes)](https://youtu.be/AI8Ub4SEEAE)
    * [Part 2 - Big Data Techniques, slide #11 (1 hour, 25 minutes)](https://youtu.be/fMmlneCaH34)
    * [Part 3 - Utilizing the Full Dataset, slides #12 ~ 19 (1 hour, 48 minutes)](https://youtu.be/BC2zkgSpHgk)
    * [Part 4 - Recommender Application, slide #20 ~ 22 (43 minutes)](https://youtu.be/4qYUdfKLfuY)
* **movie_recommendation.docx** - documentation for the source code







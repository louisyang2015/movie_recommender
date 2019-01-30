class MovieGrid{

  /**
   * @param {Element} parent_tag - html <div> tag
   * @param {Object} options - keys: "similar_movies", "history_bar"
   */
  constructor(parent_tag, options) {
    this.parent_tag = parent_tag;

    this.tmdb_poster_base_url = "https://image.tmdb.org/t/p/w342";
    this.tmdb_base_url = "https://www.themoviedb.org/movie/";

    // set defaults in "options"
    if (options.hasOwnProperty("similar_movies") === false)
      options.similar_movies = false;
    if (options.hasOwnProperty("history_bar") === false)
      options.history_bar = null;

    this.options = options;

    if (options.history_bar != null)
      options.history_bar.set_movie_grid(this);
  }

  /**
   * @param {Array} movies - array of objects. Each object has keys:
   *  "title", "movie_id", "tmdb_id", "poster_url", "rating"
   */
  render(movies) {
    let parent_tag = this.parent_tag;

    if (movies.length < 1) {
      this.render_temp_message("No result");
      return;
    }

    // remove all child nodes and start over
    while(parent_tag.firstChild)
      parent_tag.removeChild(parent_tag.firstChild);

    // go over all movies, creating tags for each one
    for (let movie of movies) {
      // <div class="movie_inner_grid">
      let movie_tag = document.createElement("div");
      movie_tag.className = "movie_inner_grid";
      parent_tag.appendChild(movie_tag);

      // movie title, there are three cases
      if (movie["poster_url"] !== "None")
      {
        // This movie has a poster
        // <a href="https://www.themoviedb.org/movie/1891">
        //         <img class="movie_poster" alt="movie title"
        //              src="https://image.tmdb.org/t/p/w342/9SKDSFbaM6LuGqG1aPWN3wYGEyD.jpg">
        // </a>
        let poster_url = this.tmdb_poster_base_url + movie["poster_url"];
        let tmdb_url = this.tmdb_base_url + movie["tmdb_id"];

        let movie_a_tag = document.createElement("a");
        movie_a_tag.setAttribute("href", tmdb_url);
        movie_tag.appendChild(movie_a_tag);

        let movie_img_tag = document.createElement("img");
        movie_img_tag.className = "movie_poster";
        movie_img_tag.setAttribute("alt", movie["title"]);
        movie_img_tag.setAttribute("src", poster_url);
        movie_a_tag.appendChild(movie_img_tag);
      }
      else if (movie["tmdb_id"] !== "None")
      {
        // This movie has a tmdb link
        // <a class="movie_link_title"
        //    href="https://www.themoviedb.org/movie/299536">
        //    Title Text xxx xxx xxx xxx xxx xxx xxx xxx </a>
        let tmdb_url = this.tmdb_base_url + movie["tmdb_id"];

        let movie_a_tag = document.createElement("a");
        movie_a_tag.className = "movie_link_title";
        movie_a_tag.setAttribute("href", tmdb_url);
        movie_a_tag.textContent = movie["title"];
        movie_tag.appendChild(movie_a_tag);
      }
      else
      {
        // This movie has only the title
        // <div class="movie_text_title">Title Text xxx xxx </div>
        let movie_title_tag = document.createElement("div");
        movie_title_tag.className = "movie_text_title";
        movie_title_tag.textContent = movie["title"];
        movie_tag.appendChild(movie_title_tag);
      }

      // Add ratings (☆☆☆☆☆) line
      let ratings_tag = document.createElement("div");
      movie_tag.appendChild(ratings_tag);
      new Stars(ratings_tag, movie);

      // Add similar movies link - if needed
      if (this.options.similar_movies)
      {
        let similar_movies_tag = document.createElement("a");
        similar_movies_tag.textContent = "similar";

        let this_obj = this;

        // handler for clicking on the similar movies link
        similar_movies_tag.addEventListener("click", function () {
          call_api(lambda_apis.similar,
            {
              "movie_id": movie["movie_id"],
              "user_id": user_id
            },
            function (response) { // handler for similar movies API return
              console.log(response);
              this_obj.render(response);

              // update history bar
              let history_bar = this_obj.options.history_bar;
              if (history_bar != null) {
                history_bar.add("Similar movies for " + movie["title"], response);
              }
            }
          );
        });

        movie_tag.appendChild(similar_movies_tag);
      }
    } // end for (let movie of movies)

    window.scrollTo(0, 0);
  }// end render(movies)


  /**
   * @param {String} message
   */
  render_temp_message(message)
  {
    let parent_tag = this.parent_tag;

    // remove all child nodes and start over
    while(parent_tag.firstChild)
      parent_tag.removeChild(parent_tag.firstChild);

    parent_tag.textContent = message;
  }
}




class Stars {
  /**
   * Each object represents five stars
   * @param {Element} parent_tag - an html tag
   * @param {Object} movie_data - has keys "movie_id" and "rating"
   * @param {boolean} hide_after_rate - if "true", hides the parent tag
   */
  constructor(parent_tag, movie_data, hide_after_rate = false) {
    this.parent_tag = parent_tag;
    this.hide_after_rate = hide_after_rate;

    // unpack "movie_data" into "movie_id" and "rating"
    this.movie_id = movie_data["movie_id"];
    this.rating = parseInt(movie_data["rating"]);

    this.star_tags = [];

    for (let i = 0; i < 5; i++)
      this.add_star(i);

    this.render_rating(this.rating);
  }

  /**
   * Adds the i-th star to "star_tags" and "parent_tag".
   * @param i - the first star is at i = 0
   */
  add_star(i) {
    let star_tag = document.createElement("span");
    star_tag.textContent = "☆";
    star_tag.className = "star";

    const this_obj = this;

    star_tag.addEventListener("mouseover", function () {
      this_obj.render_rating(i + 1);
    });

    star_tag.addEventListener("mouseleave", function () {
      this_obj.render_rating(this_obj.rating);
    });

    star_tag.addEventListener("click", function () {
      this_obj.rate(i + 1);
    });

    this.star_tags.push(star_tag);
    this.parent_tag.appendChild(star_tag);
  }

  /**
   * Render "star_tags" based on "rating".
   */
  render_rating(rating) {
    // apply solid stars to index 0 through (rating-1)
    for (let i = 0; i < rating; i++) {
      this.star_tags[i].textContent = "★";
      this.star_tags[i].style.color = "gold";
    }

    // apply empty stars to index (rating) through 4
    for (let i = rating; i < 5; i++) {
      this.star_tags[i].textContent = "☆";
      this.star_tags[i].style.color = "goldenrod";
    }
  }

  /**Event handler for the user rating a movie.*/
  rate(num_stars) {

    let this_obj = this;

    call_api(lambda_apis.rate,
      {
        "user_id": user_id,
        "op": "add_rating",
        "movie_id": this.movie_id,
        "rating": num_stars
      },
      function(response) { // handler
        if(response.error === "None") {
          this_obj.rating = num_stars;
          this_obj.render_rating(num_stars);
        }
        else
          alert("Unable to write new movie rating into database.");
      });

    if (this.hide_after_rate)
      this.parent_tag.style.visibility = "collapse";
  }
}




class HistoryBar {

  /** This is a "🡄" followed by a <span> of text.
   * @param {Element} parent_tag - an html tag
   */
  constructor(parent_tag) {
    this.history_data = [];
    this.parent_tag = parent_tag;

    // add tag for back arrow "🡄"
    let left_arrow_tag = document.createElement("a");
    // left_arrow_tag.textContent = "🡄"; // doesn't work on some browsers (MS Edge)
    left_arrow_tag.textContent = "←"; // alternative

    left_arrow_tag.className = "left_arrow";
    parent_tag.appendChild(left_arrow_tag);

    let this_obj = this;
    left_arrow_tag.addEventListener("click", function () { this_obj.back(); });

    // add <span> text tag
    this.text_tag = document.createElement("span");
    parent_tag.appendChild(this.text_tag);

    this.render();
  }


  /**
   * @param {MovieGrid} movie_grid
   */
  set_movie_grid(movie_grid) {
    this.movie_grid = movie_grid;
  }


  /**Clears the content of "history_data".*/
  clear() {
    this.history_data = [];
  }


  /**Render latest "history_data" to the tags.*/
  render() {
    let history_data = this.history_data;
    let index = history_data.length - 1;

    if (index < 1)
      this.parent_tag.style.display = "none";
    else {
      this.text_tag.textContent = history_data[index]["text"];
      this.parent_tag.style.display = "block";
    }
  }


  /**Add "text" and "data" to "history_data". */
  add(text, data) {
    this.history_data.push({"text": text, "data": data});
    this.render();
  }


  /**Go back to the previous entry in "history_data".*/
  back() {
    console.log("back");
    let history_data = this.history_data;
    if (history_data.length < 2) return;
    history_data.pop();
    this.render(); // this renders HistoryBar, but not "MovieGrid"

    let index = history_data.length - 1;
    this.movie_grid.render(history_data[index]["data"]);
  }
}




class MovieTextGrid{

  /**
   * @param {Element} parent_tag - html <div> tag
   */
  constructor(parent_tag) {
    this.parent_tag = parent_tag;
    this.child_tags = []; // 3 child <div> tags per movie

    this.tmdb_base_url = "https://www.themoviedb.org/movie/";
  }

  /**
   * @param {Array} movies - array of objects. Each object has keys:
   *  "title", "movie_id", "tmdb_id", "poster_url", "rating"
   */
  render(movies) {
    let parent_tag = this.parent_tag;

    // remove all child nodes and start over
    while(parent_tag.firstChild)
      parent_tag.removeChild(parent_tag.firstChild);

    this.child_tags = [];
    let child_tags = this.child_tags;

    // go over all movies, creating tags for each one
    for (let movie of movies) {
      let movie_tag_index = child_tags.length; // new tags will be added here

      // movie title, there are two cases
      if (movie["tmdb_id"] !== "None")
      {
        // This movie has a tmdb link
        // <a class="movie_link_title"
        //    href="https://www.themoviedb.org/movie/299536">
        //    Title Text xxx xxx xxx xxx xxx xxx xxx xxx </a>
        let tmdb_url = this.tmdb_base_url + movie["tmdb_id"];

        let movie_a_tag = document.createElement("a");
        movie_a_tag.className = "movie_link_title_smaller";
        movie_a_tag.setAttribute("href", tmdb_url);
        movie_a_tag.textContent = movie["title"];

        parent_tag.appendChild(movie_a_tag);
        child_tags.push(movie_a_tag);
      }
      else
      {
        // This movie has only the title
        // <div class="movie_text_title">Title Text xxx xxx </div>
        let movie_title_tag = document.createElement("div");
        movie_title_tag.className = "movie_text_title_smaller";
        movie_title_tag.textContent = movie["title"];

        parent_tag.appendChild(movie_title_tag);
        child_tags.push(movie_title_tag);
      }

      // Add ratings (☆☆☆☆☆) line
      let ratings_tag = document.createElement("div");
      parent_tag.appendChild(ratings_tag);
      child_tags.push(ratings_tag);

      new Stars(ratings_tag, movie);

      // Add "remove" link
      let remove_tag = document.createElement("a");
      remove_tag.textContent = "remove";
      parent_tag.appendChild(remove_tag);
      child_tags.push(remove_tag);

      let this_obj = this;

      // handler for clicking on the similar movies link
      remove_tag.addEventListener("click", function () {
        call_api(lambda_apis.rate,
          {
            "user_id": user_id,
            "op": "remove_rating",
            "movie_id": movie["movie_id"]
          },
          function (response) { // handler for rate API return
            console.log(response);

            if (response["error"] === "None")
            {
              // hide the tags associated with "movie"
              this_obj.child_tags[movie_tag_index].style.display = "none";
              this_obj.child_tags[movie_tag_index + 1].style.display = "none";
              this_obj.child_tags[movie_tag_index + 2].style.display = "none";
            }
          }
        );
      });

    } // end for (let movie of movies)

    window.scrollTo(0, 0);
  }// end render(movies)

}
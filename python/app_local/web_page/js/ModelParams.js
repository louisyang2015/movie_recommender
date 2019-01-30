class ModelParams {

  /**
   * @param {Element} parent_tag - html <div> tag
   */
  constructor(parent_tag) {
    this.parent_tag = parent_tag;
  }

  /**
   * @param {Array} movie_params - 2D array. Pass in a null to erase
   *  existing table.
   */
  render(movie_params) {
    let parent_tag = this.parent_tag;

    // remove all child nodes and start over
    while(parent_tag.firstChild)
      parent_tag.removeChild(parent_tag.firstChild);

    if(movie_params === null) return;

    // Add <h2>Model Parameters</h2>
    let h2_tag = document.createElement("h2");
    h2_tag.textContent = "Model Parameters";
    parent_tag.appendChild(h2_tag);

    let current_table = null; // current <table> tag

    // loop over "movie_params" one row at a time
    for (let i = 0; i < movie_params.length; i++) {

      // handle "__comment" rows
      if (movie_params[i][0] === "__comment") {
        // render the "__comment" row as a distinct <p> tag
        let p_tag = document.createElement("p");
        p_tag.textContent = movie_params[i][1];
        parent_tag.appendChild(p_tag);

        // start a new table if needed
        if (current_table != null) {
          current_table = document.createElement("table");
          parent_tag.appendChild(current_table);
        }
      }

      // handle standard row
      else {
        // create <table> tag if needed
        if (current_table == null) {
          current_table = document.createElement("table");
          parent_tag.appendChild(current_table);
        }

        // create <row> tag
        let current_row = document.createElement("tr");
        current_table.appendChild(current_row);

        // create one <td> for each element in movie_params[i]
        for (let col of movie_params[i]) {
          let td_tag = document.createElement("td");
          td_tag.textContent = col;
          current_row.appendChild(td_tag);
        }
      }
    } // end for (let i = 0; i < movie_params.length; i++)
  }


}
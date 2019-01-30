// Backend URLs
const lambda_apis = {
  "get_rated_movies": "http://localhost:8000/get-rated-movies",
  "rate": "http://localhost:8000/rate",
  "recommend": "http://localhost:8000/recommend",
  "search": "http://localhost:8000/search",
  "similar": "http://localhost:8000/similar"
};



// user_id is globally set
let user_id = -1;

/**Same as document.getElementById(id);*/
function get_by_id(id)
{
  document.getElementById(id);
}


/**
 * Call "url" using POST "request".
 * @param url - string
 * @param request - json object
 * @param handler - function
 * @param error_handler - function
 */
function call_api(url, request, handler, error_handler=null) {
  let http_request = new XMLHttpRequest();

  http_request.open('POST', url, true);
  http_request.send(JSON.stringify(request));

  http_request.onreadystatechange = function () {
    if (http_request.readyState === XMLHttpRequest.DONE) {

      if (http_request.status === 200) {
        let response = JSON.parse(http_request.responseText);
        handler(response);
      }
      else {
        // http_request.status != 200
        if (error_handler != null)
          error_handler();
        else
          alert('Server failed to respond.');
      }
    }
  };
}


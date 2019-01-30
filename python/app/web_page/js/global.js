// Backend URLs
const lambda_apis = {
  "get_rated_movies": "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/get-rated-movies",
  "rate": "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/rate",
  "recommend": "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/recommend",
  "search": "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/search",
  "similar": "https://fzjnokqe25.execute-api.us-west-2.amazonaws.com/beta/similar"
};



// user_id is globally set
const user_id = -1;

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


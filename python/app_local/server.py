import ast, json, socket
import get_rated_movies, rate, recommend, search, similar

port_number = 8000

api_handler = {
    "get-rated-movies": get_rated_movies.lambda_handler,
    "rate": rate.lambda_handler,
    "recommend": recommend.lambda_handler,
    "search": search.lambda_handler,
    "similar": similar.lambda_handler
}




def main():
    print("Address =", socket.gethostbyname(socket.gethostname()))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", port_number))
        sock.listen()

        print("Port number =", sock.getsockname()[1])

        while True:
            conn, addr = sock.accept()

            with conn:
                # print("Connected by", addr)
                # print("==============================")

                b_array = conn.recv(4096)
                # print(len(b_array), b_array)

                # extract the "api_url" from "b_array"
                api_url_binary = b_array.partition(b'HTTP')[0]
                api_url = api_url_binary.decode("utf-8")
                api_url = api_url.rpartition("/")[-1].strip()

                # extract json "request" from "b_array"
                last_bytes = b_array.rpartition(b'\n')[-1]
                request = ast.literal_eval(last_bytes.decode("utf-8"))

                if api_url in api_handler:
                    event = {"body": json.dumps(request)}
                    print(api_url, event["body"])
                    api_response = api_handler[api_url](event, None)

                    response = "HTTP/1.1 200 OK \n" + \
                        "Access-Control-Allow-Origin: * \n" + \
                        "Content-Type: text/html \n\n"

                    response += api_response["body"]

                    conn.sendall(response.encode())

                else:
                    response = "HTTP/1.1 400 Bad Request \n" + \
                               "Access-Control-Allow-Origin: * \n" + \
                               "Content-Type: text/html \n\n"

                    conn.sendall(response.encode())




if __name__ == "__main__":
    main()



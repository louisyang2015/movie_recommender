import json
import user_data



def lambda_handler(event, context):
    input_data = json.loads(event['body'])
    user_id = int(input_data['user_id'])
    op = input_data['op']
    movie_id = int(input_data['movie_id'])

    ratings = user_data.get_ratings(user_id)

    # handle "op": "add_rating"
    if op == "add_rating":
        rating = int(input_data['rating'])
        ratings[movie_id] = rating

    # handle "op": "remove_rating"
    elif op == "remove_rating":
        if movie_id in ratings:
            del ratings[movie_id]

    user_data.store_ratings(user_id, ratings)

    return {
        'statusCode': '200',
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        "isBase64Encoded": 'false',
        'body': json.dumps({"error": "None"})
    }





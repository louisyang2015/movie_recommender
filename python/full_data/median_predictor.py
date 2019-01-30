import cluster, movie_lens_data, my_util




def main():
    length = movie_lens_data.get_input_obj("user_ratings_test_length")

    # Sends a "median_eval" command to cluster nodes.
    if cluster.cluster_info is None:
        print("Cannot connect to cluster.")
        return

    cluster.send_command({
        "op": "distribute",
        "worker_op": "median_eval",
        "length": length
    })

    cluster.wait_for_completion()
    cluster.print_status()
    print()

    file_name = "median_eval_results.bin"
    results = cluster.merge_list_results(file_name)
    user_ids, agreements = zip(*results)

    my_util.print_rank_agreement_results(agreements, "median only")
    print('\a')


if __name__ == "__main__":
    main()


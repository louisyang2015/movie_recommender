"""
Build tasks:
    - Copy upstream files to project as needed
    - Locate the latest shared file and make all shared files the same as latest
    - Upload project files to S3 as needed

Assumptions:
    - The information in project_files.py is correct.
    - The S3 destination directories are empty.
    - All file names are unique
"""

##############################################

import boto3, os, shutil
import project_files
from project_files import upstream_files, shared_files, lambda_projects




def get_latest_up_stream_files():
    """Copy upstream files to project as needed.
    ::
        Copy only happens if upstream file is newer.
        If project file is newer than upstream, nothing happens.
    """

    for upstream_file_name, source in upstream_files.items():
        # check that "source" exists
        if os.path.exists(source) == False:
            raise Exception(source + " does not exist.")

        # get modified time for the upstream file name
        mod_time = os.path.getmtime(source)

        for project, project_file_names in lambda_projects.items():
            if upstream_file_name in project_file_names:
                # check project file for existence
                file_path = project + os.sep + upstream_file_name

                if os.path.exists(file_path) == False:
                    # Project file does not exist, copy source --> file_path
                    shutil.copy2(source, file_path)
                    print("Created:", file_path)

                else:
                    # Project file does exist, get modified time for
                    # project file
                    mod_time_project = os.path.getmtime(file_path)

                    if mod_time > mod_time_project:
                        # copy source --> file_path
                        shutil.copy2(source, file_path)
                        print("Updated:", file_path)


def update_shared_files():
    """Locate the latest shared file and make all shared
    files the same as latest."""
    for shared_file_name in shared_files:
        # For each shared_file_name, determine the location of
        # the latest version
        latest_project = None
        latest_mod_time = None

        for project, project_file_names in lambda_projects.items():
            if shared_file_name in project_file_names:
                file_path = project + os.sep + shared_file_name
                mod_time = os.path.getmtime(file_path)

                # check whether "file_path" is the latest
                if (latest_mod_time is None) or (mod_time > latest_mod_time):
                    latest_project = project
                    latest_mod_time = mod_time

        # make sure all projects have the latest shared file
        if latest_project is not None:
            latest_file_path = latest_project + os.sep + shared_file_name

            for project, project_file_names in lambda_projects.items():
                if (project != latest_project) and \
                        (shared_file_name in project_file_names):
                    file_path = project + os.sep + shared_file_name
                    mod_time = os.path.getmtime(file_path)

                    if latest_mod_time > mod_time:
                        shutil.copy2(latest_file_path, file_path)
                        print("Updated:", file_path)


def upload_file_to_s3(s3_client, full_file_name : str,
                      file_name : str, s3_key_prefix : str,
                      mod_time : float):
    """Upload object with modified time in the meta data
    :param s3_client: a reference from boto3.client("s3")
    :param full_file_name: full path of the file to upload
    :param file_name: just the file name
    :param s3_key_prefix: S3 file name is "s3_key_prefix" + "file_name"
    :param mod_time: modified time stamp from os.path.getmtime(...)
    """
    with open(full_file_name, mode="rb") as file:
        s3_client.put_object(Bucket=project_files.s3_bucket_name,
                             Key=s3_key_prefix + file_name,
                             Body=file, Metadata={"mod_time": str(mod_time)})
        print("Uploaded to S3:", s3_key_prefix + file_name)


def upload_projects_to_s3():
    """Upload project files to S3 as needed."""
    s3_client = boto3.client("s3")


    for project, project_file_names in lambda_projects.items():
        for file_name in project_file_names:
            # The S3 key prefix is either the project name, or
            # set by "project_files.s3_shared_directory"
            if (file_name in upstream_files) or (file_name in shared_files):
                s3_key_prefix = project_files.s3_shared_directory + "/"
            else:
                s3_key_prefix = project + "/"

            # get the modified time of the file
            full_file_name = project + os.sep + file_name
            mod_time = os.path.getmtime(full_file_name)

            # Determine whether an upload is needed by looking for the
            # modified time on S3
            upload_needed = False
            try:
                r = s3_client.head_object(Bucket=project_files.s3_bucket_name,
                                          Key=s3_key_prefix + file_name)
                s3_mod_time = float(r['Metadata']['mod_time'])

                if mod_time > s3_mod_time: upload_needed = True

            except:
                upload_needed = True

            if upload_needed:
                upload_file_to_s3(s3_client, full_file_name, file_name,
                                  s3_key_prefix, mod_time)





def main():
    get_latest_up_stream_files()
    update_shared_files()
    upload_projects_to_s3()
    print("All modifications have been uploaded to S3.")


if __name__ == "__main__":
    main()



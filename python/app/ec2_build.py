"""
EC2 side build script
    - Download files that have changed from S3.
    - Zip it all up and upload it back to S3 as "function.zip"

Assumptions
    - Each project is downloaded to its own directory. If the
        directory does not exist, it's created. If the directory
        does exist, it's assumed to be empty. The "zip" command
        used will compress everything in the directory. Supporting
        libraries such as Numpy needs to be set up in the project
        directories before this build script is used.
"""

meta_file = "meta"
zip_file = "function.zip"

###########################################################

import boto3, os, pickle, subprocess
import project_files
from project_files import upstream_files, shared_files, lambda_projects




def main():
    """ Download files that have changed from S3.
    Zip it all up and upload it back to S3 as "function.zip"
    """
    # retrieve meta data of files in local directory
    meta = {} # {file_name: mod_time}
    if os.path.exists(meta_file):
        with open(meta_file, mode="rb") as file:
            meta = pickle.load(file)

    s3_client = boto3.client("s3")
    s3_bucket_name = project_files.s3_bucket_name

    # keep a list of zip files uploaded to be reprinted at the end
    zip_files_uploaded = []

    for project, project_file_names in lambda_projects.items():
        # detect change
        has_changed = False

        # create directory for project if it does not exist
        if os.path.exists(project + os.sep) == False:
            os.mkdir(project)
            has_changed = True

        # go over all files for each project
        for file_name in project_file_names:
            # Determine the S3 location of "file_name"
            # The S3 key prefix is either the project name, or
            # set by "project_files.s3_shared_directory"
            if (file_name in upstream_files) or (file_name in shared_files):
                s3_key_prefix = project_files.s3_shared_directory + "/"
            else:
                s3_key_prefix = project + "/"

            full_file_name = project + os.sep + file_name

            # retrieve the S3 file's mod_time
            r = s3_client.head_object(Bucket=s3_bucket_name,
                                      Key=s3_key_prefix + file_name)
            s3_mod_time = float(r['Metadata']['mod_time'])

            # compare S3 mod_time to local mod_time to see if download is needed
            download_needed = False
            if full_file_name not in meta:
                download_needed = True
            else:
                mod_time = meta[full_file_name]
                if s3_mod_time > mod_time: download_needed = True

            if download_needed:
                s3_client.download_file(s3_bucket_name,
                                        s3_key_prefix + file_name,
                                        full_file_name)
                meta[full_file_name] = s3_mod_time
                print("Downloaded:", full_file_name)
                has_changed = True

        # end: for file_name in project_file_names:

        # save meta information
        with open(meta_file, mode="wb") as file:
            pickle.dump(meta, file)

        if has_changed:
            # remove "zip_file" if it exists
            project_zip_file = project + os.sep + zip_file
            if os.path.exists(project_zip_file):
                os.remove(project_zip_file)

            # issue zip command, which looks like:
            # cd project_name; zip -r function.zip .; cd ..
            zip_command = "cd " + project + "; zip -r " + zip_file + " .; cd .."

            subprocess.call(zip_command, shell=True)

            # upload zip file
            s3_client.upload_file(project_zip_file, s3_bucket_name,
                                  project_zip_file)
            zip_file_s3_url = "https://s3.amazonaws.com/" + s3_bucket_name \
                              + "/" + project_zip_file

            print("Uploaded:", project_zip_file)
            print("To:", zip_file_s3_url)
            zip_files_uploaded.append(zip_file_s3_url)

        else:
            print("No change for project ", project)

    # end for project, project_file_names in lambda_projects.items():
    if len(zip_files_uploaded) > 0:
        print("\nZip files uploaded:")

        for zip_file_name in zip_files_uploaded:
            print("   ", zip_file_name)



if __name__ == "__main__":
    main()



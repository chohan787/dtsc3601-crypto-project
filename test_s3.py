import boto3

session = boto3.Session(profile_name="dtsc3601-crypto")
s3 = session.client("s3")

bucket_name = "dtsc3601-cryptobros-project-879022683465-us-east-2-an"

s3.put_object(
    Bucket=bucket_name,
    Key="python-test.txt",
    Body="This file was uploaded using Python and boto3."
)

print("Upload successful")
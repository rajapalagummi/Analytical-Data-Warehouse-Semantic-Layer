import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

GLUE_ROLE = os.environ.get("AWS_GLUE_ROLE_ARN")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "enterprise-dw-glue")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
JOB_NAME = "enterprise-dw-transformation"

TRANSFORMATION_SCRIPT = """
import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_path', 'target_path'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

raw = spark.read.parquet(args['source_path'])

raw = raw.withColumn('loaded_at', F.current_timestamp())

window = Window.partitionBy('category').orderBy(F.col('revenue').desc())
transformed = (
    raw
    .withColumn('revenue_rank', F.rank().over(window))
    .withColumn('category_total', F.sum('revenue').over(Window.partitionBy('category')))
    .withColumn('revenue_pct', F.col('revenue') / F.col('category_total'))
    .filter(F.col('revenue') > 0)
)

schema_tests = [
    transformed.filter(F.col('revenue').isNull()).count() == 0,
    transformed.filter(F.col('category').isNull()).count() == 0,
    transformed.count() > 0,
]
assert all(schema_tests), "Schema validation failed"

transformed.write.mode('overwrite').partitionBy('category').parquet(args['target_path'])
job.commit()
"""


def upload_script(script_content, bucket, key):
    s3 = boto3.client('s3', region_name=REGION)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    s3.put_object(Bucket=bucket, Key=key, Body=script_content.encode())
    return f"s3://{bucket}/{key}"


def create_glue_job(script_s3_path):
    glue = boto3.client('glue', region_name=REGION)
    try:
        glue.delete_job(JobName=JOB_NAME)
    except glue.exceptions.EntityNotFoundException:
        pass

    response = glue.create_job(
        Name=JOB_NAME,
        Role=GLUE_ROLE,
        Command={
            'Name': 'glueetl',
            'ScriptLocation': script_s3_path,
            'PythonVersion': '3',
        },
        DefaultArguments={
            '--source_path': f's3://{S3_BUCKET}/raw/',
            '--target_path': f's3://{S3_BUCKET}/transformed/',
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics': '',
        },
        GlueVersion='4.0',
        NumberOfWorkers=2,
        WorkerType='G.1X',
        MaxRetries=1,
        Timeout=60,
    )
    return response


def start_glue_job(source_path=None, target_path=None):
    glue = boto3.client('glue', region_name=REGION)
    args = {}
    if source_path:
        args['--source_path'] = source_path
    if target_path:
        args['--target_path'] = target_path

    response = glue.start_job_run(JobName=JOB_NAME, Arguments=args)
    run_id = response['JobRunId']
    print(f"Glue job started: {run_id}")
    return run_id


def get_job_status(run_id):
    glue = boto3.client('glue', region_name=REGION)
    response = glue.get_job_run(JobName=JOB_NAME, RunId=run_id)
    return response['JobRun']['JobRunState']


if __name__ == "__main__":
    print("Uploading Glue ETL script to S3...")
    script_path = upload_script(
        TRANSFORMATION_SCRIPT,
        S3_BUCKET,
        f"scripts/{JOB_NAME}.py"
    )
    print(f"Script uploaded: {script_path}")

    print("Creating Glue job definition...")
    create_glue_job(script_path)
    print(f"Glue job created: {JOB_NAME}")

    print("Starting Glue ETL job...")
    run_id = start_glue_job()
    print(f"Job run ID: {run_id}")
    print(f"Monitor at: https://console.aws.amazon.com/glue/home#/etl/jobs/runs/{JOB_NAME}")

from google.cloud import bigquery

bqclient = bigquery.Client(project='valohai-dev')

# Download query results.
query_string = """
SELECT
CONCAT(
    'https://stackoverflow.com/questions/',
    CAST(id as STRING)) as url,
view_count
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE tags like '%google-bigquery%'
ORDER BY view_count DESC
"""

df = (
    bqclient.query(query=query_string)
    .result()
    .to_dataframe()
)
print(df.head())

df.to_csv('/valohai/outputs/results.csv')

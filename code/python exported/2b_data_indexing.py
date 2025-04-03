# %%
import os  # For interacting with the file system
from elasticsearch import helpers, Elasticsearch
import csv
import pandas as pd

# %%
# Retrieve authentication information for Elasticsearch
elastic_host = "https://localhost"
elastic_port = "9200"
elastic_user = "admin"
elastic_password = "motdepasse"
elastic_ca_path = "C:\\elasticsearch-8.15.2\\config\\certs\\http_ca.crt"

# Connect to Elasticsearch
es = Elasticsearch(
    hosts=[f"{elastic_host}:{elastic_port}"],
    basic_auth=(elastic_user, elastic_password),
    ca_certs=elastic_ca_path,
    verify_certs=True
)
print(es.info())

# Check connection
if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Failed to connect to Elasticsearch")

# %%
index_name = "network_flows_fan_encoded_final"
csv_file_name = "csv_files/final_features_flows.csv"

# %%
# retrieve the df from the csv
df = pd.read_csv(csv_file_name)

# Removing the encoded original cols
df = df.drop(columns=['bidirectional_first_seen_ms', 'bidirectional_last_seen_ms','src_port','dst_port','src_ip','dst_ip'])

# %%
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Index {index_name} deleted.")
es.indices.create(index=index_name)
print(f"Index {index_name} created with specified mapping.")

# %%
# Prepare data for indexing in ES
actions = [
    {
        "_index": target_index,
        "_source": row.to_dict()
    }
    for _, row in df.iterrows()
]

# Indexing with batch in ES
batch_size = 50
for i in range(0, len(actions), batch_size):
    helpers.bulk(es, actions[i:i + batch_size])

print(f"Indexing in '{target_index}' finished.")



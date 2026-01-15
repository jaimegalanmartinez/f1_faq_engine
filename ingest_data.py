import os
import chardet
from sentence_transformers import SentenceTransformer
import pandas as pd
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.init import Auth
from dotenv import load_dotenv

def read_csv_with_encoding(file_path):
    with open(file_path, 'rb') as f:
         # Read a chunk to guess encoding (reading whole file is slower) 10 KB
        result = chardet.detect(f.read())
    encoding = result['encoding']
    confidence = result["confidence"]
    print(f"Reading {os.path.basename(file_path)} as {encoding} ({confidence:.2f} confidence)")
    return pd.read_csv(file_path, encoding=encoding)

file_map = {
    'races': 'data/races.csv',
    'results': 'data/results.csv',
    'drivers': 'data/drivers.csv',
    'constructors': 'data/constructors.csv',
    'circuits': 'data/circuits.csv'
}

print("Loading CSVs with dynamic encoding detection...")
dataframes = {}
for name, path in file_map.items():
    if os.path.exists(path):
        dataframes[name] = read_csv_with_encoding(path)

races = dataframes.get('races')
results = dataframes.get('results')
drivers = dataframes.get('drivers')
constructors = dataframes.get('constructors')
circuits = dataframes.get('circuits')

# Merge Data
df = results.merge(races, on='race_id')
df = df.merge(drivers, on='driver_id')
df = df.merge(constructors, on='constructor_id')
df = df.merge(circuits, on='circuit_id')


# Clean and prepare dataframe
df.rename(columns={'name_x': 'constructor_name','givenName': 'driver_name',
                   'familyName':'driver_surname','nationality_x': 'driver_nationality',
                   'nationality_y': 'constructor_nationality', 'name_y':'circuit_name'}, inplace=True)
df = df[['season', 'race_name', 'driver_name', 'driver_surname', 'constructor_name', 'position_order', 'points', 'round', 'date', 'circuit_name']]
df = df.fillna('')


# Load Open Source Model locally
model = SentenceTransformer('BAAI/BGE-M3')

# Prepare data
data_objects = []
for _, row in df.iterrows():
    # Ensuring strict types to prevent Weaviate errors
    text = (f"On {str(row['date'])}, during the {int(row['season'])} season, at the {str(row['race_name'])} celebrated in {str(row['circuit_name'])} (Round {int(row['round'])}), "
            f"driver {str(row['driver_name'])} {str(row['driver_surname'])} driving for {str(row['constructor_name'])} "
            f"finished in position {str(row['position_order'])} with {float(row['points'])} points.")
    
    vector = model.encode(text).tolist()

    data_objects.append({
        "content": text,
        "year": int(row["season"]),
        "vector": vector  # <--- We send the pre-calculated vector
    })

print(f"Created {len(data_objects)} documents.")

# Load environment variables from .env file
load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
cluster_url=weaviate_url,
auth_credentials=Auth.api_key(weaviate_api_key),
)

try:
    print(f'Weaviate client is ready:"{client.is_ready()}')
    # Create collection without a vectorizer (vectorizer_config=none)
    if client.collections.exists("F1Context"):
        client.collections.delete("F1Context")

    f1_collection = client.collections.create(
        name="F1Context",
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
        ],
    )

    # Upload Data with the vectors
    with f1_collection.batch.dynamic() as batch:
        for obj in data_objects:
            batch.add_object(
                properties={
                    "content": obj["content"],
                    "year": obj["year"]
                },
                vector=obj["vector"] # Manually inject the vector
            )

    print("Ingestion Complete!")
finally:
    client.close()


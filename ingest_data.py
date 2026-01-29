import os
import sys
import chardet
from sentence_transformers import SentenceTransformer
import pandas as pd
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.init import Auth
from dotenv import load_dotenv


def read_csv_with_encoding(file_path):
    """Read CSV with automatic encoding detection."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    confidence = result["confidence"]
    print(
        f"Reading {
            os.path.basename(file_path)} as {encoding} ({
            confidence:.2f} confidence)")
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
print("Merging data...")
df = results.merge(races, on='race_id')
df = df.merge(drivers, on='driver_id')
df = df.merge(constructors, on='constructor_id')
df = df.merge(circuits, on='circuit_id')


# Clean and prepare dataframe
df.rename(columns={'name_x': 'constructor_name',
                   'givenName': 'driver_name',
                   'familyName': 'driver_surname',
                   'nationality_x': 'driver_nationality',
                   'nationality_y': 'constructor_nationality',
                   'name_y': 'circuit_name'
                   }, inplace=True)
df = df[['season', 'race_name', 'driver_name', 'driver_surname',
         'constructor_name', 'position_order', 'points', 'round',
         'date', 'circuit_name']]


# Fill remaining NaN with empty strings
df = df.fillna('')

# Reset index to ensure sequential indexing
df = df.reset_index(drop=True)

# Load Open Source Model locally
print("Loading embedding model (BAAI/bge-m3)...")
model = SentenceTransformer('BAAI/BGE-M3')

# Prepare data
print("Creating text descriptions...")
texts = []
for _, row in df.iterrows():
    position = int(row['position_order'])
    points = float(row['points'])

    if position == 1:
        position_text = "won the race, finishing in first place"
    elif position == 2:
        position_text = "finished in second place"
    elif position == 3:
        position_text = "finished in third place, securing a podium finish"
    else:
        if 11 <= (position % 100) <= 13:
            suffix = "th"  # 11th, 12th, 13th
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
        position_text = f"finished in {position}{suffix} place"

    if points > 0:
        points_text = f"scoring {points} championship points"
    else:
        points_text = "outside the points"

    # Format 1: For winners and podium (emphasize achievement)
    if position <= 3:
        text = (
            f"{row['driver_name']} {row['driver_surname']} "
            f"{position_text} "
            f"at the {row['race_name']} in {int(row['season'])}, "
            f"racing for {row['constructor_name']} at {row['circuit_name']}. "
            f"This was Round {int(row['round'])} on {row['date']}, {points_text}."
        )
    # Format 2: Not in podium
    else:
        text = (
            f"In the {int(row['season'])} {row['race_name']} "
            f"at {row['circuit_name']} (Round {int(row['round'])}), "
            f"{row['driver_name']} {row['driver_surname']} "
            f"driving for {row['constructor_name']} "
            f"{position_text}, {points_text} on {row['date']}."
        )
    texts.append(text)


print(f"Encoding {len(texts)} texts. with batch processing...")
vectors = model.encode(texts, show_progress_bar=True, batch_size=32)


print("Preparing data objects...")
data_objects = []
for idx, (_, row) in enumerate(df.iterrows()):
    data_objects.append({
        "content": texts[idx],
        "year": int(row["season"]),
        "race_name": str(row['race_name']),
        "driver_name": f"{str(row['driver_name'])} {str(row['driver_surname'])}",
        "constructor_name": str(row["constructor_name"]),
        "circuit_name": str(row["circuit_name"]),
        "position": int(row["position_order"]),
        "points": float(row["points"]),
        "round": int(row["round"]),
        "vector": vectors[idx].tolist()  # <--- We send the pre-calculated vector
    })

print(f"Created {len(data_objects)} documents.")

force_recreate = "--force" in sys.argv
# Load environment variables from .env file
load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

if not weaviate_url or not weaviate_api_key:
    raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")

# Connect to Weaviate Cloud
print("Connecting to Weaviate Cloud...")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

try:
    print(f'Weaviate client is ready:"{client.is_ready()}')

    # Delete existing collection
    if client.collections.exists("F1Context"):
        if not force_recreate:
            raise RuntimeError(
                "Collection 'F1Context' already exists. "
                "Pass --force to delete and recreate it."
            )
        print("Deleting existing F1Context collection...")
        client.collections.delete("F1Context")

    f1_collection = client.collections.create(
        name="F1Context",
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
            Property(name="race_name", data_type=DataType.TEXT),
            Property(name="driver_name", data_type=DataType.TEXT),
            Property(name="constructor_name", data_type=DataType.TEXT),
            Property(name="circuit_name", data_type=DataType.TEXT),
            Property(name="position", data_type=DataType.INT),
            Property(name="points", data_type=DataType.NUMBER),
            Property(name="round", data_type=DataType.INT),
        ],
        # Optimize vector index for better retrieval
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=200,
                max_connections=64
            )
        )
    )

    # Upload Data with the vectors
    print(f"\nIngesting {len(data_objects)} objects...")
    failed_count = 0

    with f1_collection.batch.dynamic() as batch:
        for obj in data_objects:
            batch.add_object(
                properties={
                    "content": obj["content"],
                    "year": obj["year"],
                    "race_name": obj["race_name"],
                    "driver_name": obj["driver_name"],
                    "constructor_name": obj["constructor_name"],
                    "circuit_name": obj["circuit_name"],
                    "position": obj["position"],
                    "points": obj["points"],
                    "round": obj["round"]
                },
                vector=obj["vector"]  # Manually inject the vector
            )

    # Check for failures after the batch completes
    if hasattr(
            f1_collection.batch,
            'failed_objects') and f1_collection.batch.failed_objects:
        failed_count = len(f1_collection.batch.failed_objects)
        print(f"\n❌ {failed_count} objects failed during ingestion:")
        for i, failure in enumerate(f1_collection.batch.failed_objects[:5], 1):
            print(f"  [{i}] {failure}")
        if failed_count > 5:
            print(f"  ... and {failed_count - 5} more")

    success_count = len(data_objects) - failed_count
    print(f"\n✅ Ingestion complete: {success_count} succeeded, {failed_count} failed")

finally:
    client.close()
    print("\nConnection closed.")

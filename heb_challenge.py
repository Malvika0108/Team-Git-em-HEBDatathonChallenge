from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import json
import pandas as pd
import sys

with open('/content/products.json', 'r') as file:
    products = json.load(file)

with open('/content/labels_synth_train.json', 'r') as file:
    labelsTrain = json.load(file)

with open('/content/queries_synth_train.json', 'r') as file:
    queriesTrain = json.load(file)

print(queriesTrain)

print(labelsTrain)

print(products)

labelsTrain_df = pd.DataFrame(labelsTrain)
queriesTrain_df = pd.DataFrame(queriesTrain)
products_df = pd.DataFrame(products)

mergedData = labelsTrain_df.merge(queriesTrain_df, on="query_id").merge(products_df, on="product_id")

if 'product_text' not in mergedData.columns:
    print("Creating 'product_text' column for mergedData...")
    def create_product_text(row):
        return (
            f"Title: {row['title']}. "
            f"Brand: {row['brand']}. "
            f"Category: {row['category_path']}. "
            f"Description: {row['description']}. "
            f"Ingredients: {row['ingredients']}"
        )
    mergedData['product_text'] = mergedData.apply(create_product_text, axis=1)
    print("Done creating 'product_text' for mergedData.")

# Ensure products_df also has the 'product_text' column
if 'product_text' not in products_df.columns:
    print("Creating 'product_text' column for products_df...")
    products_df['product_text'] = products_df.apply(create_product_text, axis=1)
    print("Done creating 'product_text' for products_df.")

max_relevance = mergedData['relevance'].max()
mergedData['norm_relevance'] = mergedData['relevance'] / max_relevance

# Create InputExample objects
train_examples = []
for index, row in mergedData.iterrows():
    train_examples.append(
        InputExample(
            texts=[row['query'], row['product_text']],
            label=row['norm_relevance']
        )
    )

print(f"Created {len(train_examples)} training examples.")

model = SentenceTransformer('all-MiniLM-L6-v2')

train_loss = losses.CosineSimilarityLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of steps

print("Starting model training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='./heb_retriever_model' # Path to save the trained model
)

print("Training complete. Model saved to './heb_retriever_model'.")

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('./heb_retriever_model')
unique_products = products_df.drop_duplicates(subset=['product_id'])
product_texts = unique_products['product_text'].tolist()
product_ids = unique_products['product_id'].tolist()

print(f"Encoding {len(product_texts)} products...")
product_embeddings = model.encode(product_texts, convert_to_tensor=True, show_progress_bar=True)
print(f"Product embeddings shape: {product_embeddings.shape}")


user_query = "healthy snacks for kids"

print(f"\nSearching for: '{user_query}'")

query_embedding = model.encode(user_query, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, product_embeddings, top_k=5)

hits = hits[0]

print("Top 5 results:")
for hit in hits:
    product_index = hit['corpus_id']
    score = hit['score']

    product_id = product_ids[product_index]
    product_title = unique_products.iloc[product_index]['title']

    print(f"  - Score: {score:.4f} \t ID: {product_id} \t Title: {product_title}")

print("Loading trained model...")
model = SentenceTransformer('./heb_retriever_model')

if 'product_text' not in products_df.columns:
    print("Creating 'product_text' column...")
    def create_product_text(row):
        return (
            f"Title: {row['title']}. "
            f"Brand: {row['brand']}. "
            f"Category: {row['category_path']}. "
            f"Description: {row['description']}. "
            f"Ingredients: {row['ingredients']}"
        )
    products_df['product_text'] = products_df.apply(create_product_text, axis=1)

unique_products = products_df.drop_duplicates(subset=['product_id'])
product_texts = unique_products['product_text'].tolist()
product_ids_list = unique_products['product_id'].tolist() # This list maps index -> product_id

print(f"Encoding {len(product_texts)} products for the vector index...")
product_embeddings = model.encode(product_texts, convert_to_tensor=True, show_progress_bar=True)
print("Product encoding complete.")

print("Loading test queries...")
try:
    with open('/content/queries_synth_test.json', 'r') as file:
        test_queries = json.load(file)
except FileNotFoundError:
    print("Error: /content/queries_synth_test.json not found.")
    # sys.exit("Stopping script.")
    test_queries = [] # Avoid crashing the rest of the example

TOP_K_RESULTS = 30

submission_list = []
print(f"Generating rankings for {len(test_queries)} test queries...")

for test_query in test_queries:
    query_id = test_query['query_id']
    query_text = test_query['query']

    query_embedding = model.encode(query_text, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, product_embeddings, top_k=TOP_K_RESULTS)
    hits = hits[0]

    for rank, hit in enumerate(hits, start=1):
        product_index = hit['corpus_id']
        product_id = product_ids_list[product_index]

        submission_entry = {
            "query_id": query_id,
            "rank": rank,
            "product_id": product_id
        }
        submission_list.append(submission_entry)

print(f"Generated {len(submission_list)} total ranking entries.")

print(len(submission_list))

print("Saving submission.json...")
with open('submission.json', 'w') as f:
    # Use indent=2 for a readable, formatted file
    json.dump(submission_list, f, indent=2)

print("\nAll done! 'submission.json' has been created.")


#https://huggingface.co/blog/image-similarity
#https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_similarity.ipynb#scrollTo=u2yJIlPbgGGh

from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image

import torchvision.transforms as T

# Loading a base model to compute embeddings. "Embeddings" encode the semantic information of images. 
# To compute the embeddings from the images, we'll use a vision model that has some understanding 
# of how to represent the input images in the vector space. 
# This type of models is also commonly referred to as image encoders.

# For loading the model, we leverage the AutoModel class. 
# It provides an interface for us to load any compatible model checkpoint from the Hugging Face Hub. 
# Alongside the model, we also load the processor associated with the model for data preprocessing.

model_name = "google/vit-base-patch16-224-in21k"
model = AutoModel.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Data transformation chain.
transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((256 / 224) * feature_extractor.size["height"])),
        T.CenterCrop(feature_extractor.size["height"]),
        T.ToTensor(),
        #T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ]
)

import torch

# Extract the embeddings from the candidate images (candidate_subset) storing them in a matrix.
# Modified to read files form a folder

def extract_embeddings(image_paths, feature_extractor, model, batch_size=32):
    device = model.device
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_images = []
        for image_path in image_paths[i:i+batch_size]:
            image = Image.open(image_path).convert("RGB")
            image = transformation_chain(image)
            batch_images.append(image)
        batch_images = torch.stack(batch_images).to(device)
        with torch.no_grad():
            #features = feature_extractor(images=batch_images, return_tensors="pt")
            embeddings.append(model(**batch_images).last_hidden_state[:, 0].cpu())
    embeddings = torch.cat(embeddings)
    return embeddings

batch_size = 24
device = "cuda" if torch.cuda.is_available() else "cpu"
#extract_fn = extract_embeddings(model.to(device))
#candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=24)

import glob, os
view_foder = '/home/ec2-user/3d/matching/2dviews'
view_paths = os.path.join(view_foder, "*")
view_image_names = glob.glob(os.path.join(view_foder, "*"))
   
candidate_subset_emb = extract_embeddings(view_image_names, feature_extractor, model.to(device),batch_size=2)
print (len(candidate_subset_emb), len(candidate_subset_emb[0]))

import numpy as np
all_candidate_embeddings = np.array(candidate_subset_emb)
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

# use cosine similarity to compute the similarity score in between two embedding vectors
def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()

def fetch_similar(image, top_k=5):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    image_transformed = transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    # Comute the embedding.
    with torch.no_grad():
        query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    sim_scores = compute_scores(all_candidate_embeddings, query_embeddings)
    similarity_mapping = dict(zip(candidate_ids, sim_scores))
    
    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    similars = list(similarity_mapping_sorted.keys())[:top_k]

    return similars



# extract feature from input image
input_image_file = "10_dining_table_cropped_640_480.png"
input_image = Image.open(input_image_file).convert("RGB")
similar_results = fetch_similar(input_image,5)
print(similar_results)




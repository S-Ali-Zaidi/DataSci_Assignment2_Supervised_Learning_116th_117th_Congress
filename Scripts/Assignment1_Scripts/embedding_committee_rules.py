import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# this script is used to embed committee rules text using OpenAI's text-embedding-3-large model
# because no token limits are imposed on committee rules text, we can embed all the text in one go with this script

# per openai docs, it's good practice to load the api key from a .env file
load_dotenv()
# leaving this line commented out to avoid accidental execution and wasting API credits. Uncomment when running the script
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# input and output paths for committee rules text and embeddings
input_csv = 'Data/Main_Data/Committees.csv'  
output_folder = 'Data/embeddings/committees'  
os.makedirs(output_folder, exist_ok=True)

# We're going to use the text-embedding-3-large model for this task
# note that i set a very large batch size here, but in reality, there are only 21 committees, so the batch size will take care of them all within one request
model = "text-embedding-3-large"  
batch_size = 500

# load csv with committee rules text
df = pd.read_csv(input_csv)

# And then here is our function that will send the text to the OpenAI API and save the embeddings
def generate_and_save_embeddings(df: pd.DataFrame, model: str, batch_size: int, output_folder: str) -> None:

    # loop through data in batches
    for batch_start in range(0, len(df), batch_size):
        batch_end = batch_start + batch_size
        batch = df.iloc[batch_start:batch_end]
        
        # extract committee rules text and committee name
        batch_texts = batch['Rules'].tolist()
        batch_committees = batch['Committee'].tolist()

        # embed
        response = client.embeddings.create(input=batch_texts, model=model)
        
        # save each embedding, we are saving the embeddings as numpy arrays to worth with later on in our analysis
        for i, embedding_object in enumerate(response.data):
            embedding = embedding_object.embedding  
            committee_name = batch_committees[i]
            filename = f"{committee_name}.npy"
            filepath = os.path.join(output_folder, filename)
            np.save(filepath, embedding)
            print(f"Saved embedding for {committee_name} to {filename}")

        # General precaution i leave in my scripts to avoid hitting the API rate limit. not needed here since we're only making one request in a sinle batch
        time.sleep(0.5)


generate_and_save_embeddings(df, model, batch_size, output_folder)

print("Done!")
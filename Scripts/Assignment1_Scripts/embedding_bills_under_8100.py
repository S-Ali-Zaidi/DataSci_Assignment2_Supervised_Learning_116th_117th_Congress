import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# this script is used to embed bills text using OpenAI's text-embedding-3-large model
# It only processes bills with fewer than 8,100 tokens to avoid hitting the token limit of the OpenAI API, so the logic is fairly similar to the embedding_committee_rules.py script.

# per openai docs, it's good practice to load the api key from a .env file
load_dotenv()
print("API Key:", os.environ.get("OPENAI_API_KEY"))
# leaving this line commented out to avoid accidental execution and wasting API credits. Uncomment when running the script
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths for input CSV, cleaned text files, and output embeddings
input_csv = 'Data/Main_Data/Bills.csv'
txt_cleaned_folder = 'Data/bill_texts/txt_cleaned'
output_folder = 'Data/embeddings/bills'
os.makedirs(output_folder, exist_ok=True)

# Again, we're using the text-embedding-3-large model
# this time, since we are processing thousands of bills, we will use a batch size of 500 to avoid hitting the token limit.
# this should be modified in accordance with the rate limit tier you are given by OpenAI.
model = "text-embedding-3-large"
batch_size = 500

# Load CSV a
df = pd.read_csv(input_csv)

# Check to see if 'embedded?' and 'tokens_used' columns exist in the DataFrame
# this was added in case the embedding process was interrupted and we need to resume from where we left off
# otherwise we'd waste API credits by re-embedding the same bills
if 'embedded?' not in df.columns:
    df['embedded?'] = ""
if 'tokens_used' not in df.columns:
    df['tokens_used'] = ""

# Filter the DataFrame for bills with fewer than 8,100 tokens and not yet embedded
df_under_8100 = df[(df['Token Count'] < 8100) & (df['embedded?'] != "yes")]

# this function again is almost identical to the one in embedding_committee_rules.py, with modifications to handle bill text files rather that text stored in a CSV
def generate_and_save_embeddings(df: pd.DataFrame, model: str, batch_size: int, output_folder: str) -> None:
    # Logic for processing bills in batches
    for batch_start in range(0, len(df_under_8100), batch_size):
        batch_end = batch_start + batch_size
        batch = df_under_8100.iloc[batch_start:batch_end]
        
        # variables to store text and bill numbers for each batch
        batch_texts = []
        batch_bill_numbers = []
        
        # Loop through each bill in the batch. Attach .txt to the bill number to get the corresponding file name from the txt_cleaned folder
        for _, row in batch.iterrows():
            bill_number = row['Bill Name']
            file_path = os.path.join(txt_cleaned_folder, f"{bill_number}.txt")
            
            # Only process if the file exists and flag if it doesn't
            # we are also ensuring that all texts are being read in utf-8 encoding for consistency
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    batch_texts.append(content)
                    batch_bill_numbers.append(bill_number)
            else:
                print(f"Text file for {bill_number} not found. Skipping.")
        
        if not batch_texts:
            continue  # this is to avoid empty batches where no text files were found (because they were alraady embedded in a previous run)

        # Perform embedding
        response = client.embeddings.create(input=batch_texts, model=model)
        
        # Save embeddings and mark as embedded
        for i, embedding_object in enumerate(response.data):
            embedding = embedding_object.embedding  
            bill_number = batch_bill_numbers[i]
            filename = f"{bill_number}.npy"
            filepath = os.path.join(output_folder, filename)
            np.save(filepath, embedding)
            print(f"Saved embedding for {bill_number} to {filename}")
            
            # Marking the bill as embedded to avoid re-embedding in future runs
            df.loc[df['Bill Name'] == bill_number, 'embedded?'] = "yes"

        # Capturing the total tokens used in the response from openai -- this was just for me to see if the tiktoken library was accurate in counting tokens
        tokens_used = response.usage.total_tokens
        for bill_number in batch_bill_numbers:
            df.loc[df['Bill Name'] == bill_number, 'tokens_used'] = tokens_used

        # saving the updated CSV after each batch
        df.to_csv(input_csv, index=False)
        print("CSV updated with embedded status and tokens used for the current batch.")
        
        # General precaution i leave in my scripts to avoid hitting the API rate limit. 
        time.sleep(0.5)

# run the function to generate and save embeddings
generate_and_save_embeddings(df, model, batch_size, output_folder)

print("Done with embeddings for texts under 8,100 tokens!")
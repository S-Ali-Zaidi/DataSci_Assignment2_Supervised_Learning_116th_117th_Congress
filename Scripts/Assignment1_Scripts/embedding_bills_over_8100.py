import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# this script is used to embed bills whose texts we identified as taking up more than 8,100 tokens, the maximum token limit for the OpenAI API
# We split the text into chunks of 6,000 tokens each to ensure we stay within the token limit
# the embeddings are then averaged to get a single embedding for the entire bill

# per openai docs, it's good practice to load the api key from a .env file
load_dotenv()
# leaving this line commented out to avoid accidental execution and wasting API credits. Uncomment when running the script
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths for input CSV, cleaned text files, and output embeddings
input_csv = 'Data/Main_Data/Bills.csv'
txt_cleaned_folder = 'Data/bill_texts/txt_cleaned'
output_folder = 'Data/embeddings/bills'
os.makedirs(output_folder, exist_ok=True)

# OpenAI model
model = "text-embedding-3-large"

# Load CSV
df = pd.read_csv(input_csv)

# Again, this is here as  a precaution to avoid re-embedding bills that have already been processed in a previous incompleted run
if 'embedded?' not in df.columns:
    df['embedded?'] = ""
if 'tokens_used' not in df.columns:
    df['tokens_used'] = ""

# Filter to only focus on bills without a "yes" in the 'embedded?' column
# the assumption is that this is being ran AFTER the embedding_bills_under_8100.py script
# so all logic here is meant to only be ran on bills above the 8,100 token limit

df_to_embed = df[df['embedded?'] != "yes"]

# These two functions are used to split the text into chunks of 6,000 tokens each (with the final chunk potentially being smaller)
# they are helper functions to be used in the main function below
def calculate_chunks(text, chunk_count):
    chunk_size = len(text) // chunk_count
    return [text[i * chunk_size: (i + 1) * chunk_size] for i in range(chunk_count)]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# this is the main function that will process the bills in chunks and save the averaged embeddings
# it relies on the helper functions above and uses tiktoken to count the number of tokens in each chunk.
# if tiktoken tells us that a chunk is too large, we skip the bill and move on to the next one
# I started with a division by 8000, and as some bills continued to have chunks still too large, i decreased the limit in increments of 500
# eventually, on a fifth pass of running this script, i finally got all the largest texts emebedded using a division by 6000 to get all chunks under 8100 tokens
def generate_and_save_embeddings(df: pd.DataFrame, model: str, output_folder: str) -> None:
    
    for _, row in df_to_embed.iterrows():
        bill_number = row['Bill Name']
        token_count = row['Token Count']

        # Calculate chunk count based on token count divided by some number. I started with 8000 and decreased in increments of 500 until the final pass used division by 6000
        # Frankly, it might have been better to have just ran the script over on all the bills again with the final division by 6000, but I didn't want to waste API credits
        # Additionally, i wanted each chunk embedding to be representative of as much information as possible, to prevent some embedding from swinging wildly in one direction and effecting the average
        # My theory is that is ensures that the final embedding is more representative of the entire bill and comparable to the other embeddings comprised of a single chunk
        chunk_count = int(np.ceil(token_count / 6000))
        print(f"\nProcessing {bill_number} with token count {token_count}")
        print(f"Calculated number of chunks/batch size: {chunk_count}")

        file_path = os.path.join(txt_cleaned_folder, f"{bill_number}.txt")

        # Only process if the text file exists, otherwise skip and notify me of the missing file
        if not os.path.exists(file_path):
            print(f"Text file for {bill_number} not found. Skipping.")
            continue
        
        # Load text content, and again we ensure utf-8 encoding for consistency. Probably not necessary here, but good practice. 
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split text into chunks using the functions we defined above
        chunks = calculate_chunks(content, chunk_count)
        
        # Again, prior to sending the chunks for embedding to the openai API, we check if any chunk is too large, and if so, ensure it is skipped in the latest run of this script
        # setting a division by 6000 from the get-go would have allowed the script to cover all bills in one go, but i wanted avoid this for the reasons mentioned above
        chunk_texts = []
        valid_chunks = True
        for idx, chunk in enumerate(chunks):
            chunk_token_count = num_tokens_from_string(chunk, "cl100k_base")
            print(f"Chunk {idx + 1} token count: {chunk_token_count}")
            
            if chunk_token_count > 8100:
                print(f"Chunk {idx + 1} exceeds 8100 tokens, skipping {bill_number}")
                valid_chunks = False
                break
            chunk_texts.append(chunk)

        if not valid_chunks:
            continue  # Skip to next bill if any chunk is too large
        
        # Perform embedding if all chunks are valid, since we are processing each bill one at a time, rather than in batches as we did with the under 8100 token bills
        print(f"Sending {len(chunk_texts)} chunks for {bill_number} as a batch.")
        response = client.embeddings.create(input=chunk_texts, model=model)
        
        # simple function to extract the embeddings from the response from OpenAI
        # and then to average all embeddings for this bill to get a single embedding for the entire bill that can be saved and compared with other bills under 8100 tokens. 
        embeddings = [embedding_object.embedding for embedding_object in response.data]
        averaged_embedding = np.mean(embeddings, axis=0)
        
        # attach .npy to the bill number when saving the embedding to the output folder
        filename = f"{bill_number}.npy"
        filepath = os.path.join(output_folder, filename)
        np.save(filepath, averaged_embedding)
        print(f"Averaged embedding saved for {bill_number} to {filename}")

        # Record tokens used for the batch and update the CSV with the bill marked as embedded
        # This allowed me to compare how many tokens were used in this chunked embedding process compared to tiktoken's count of the entire bill text.
        # If i saw that the token count was close to the count of the entire text, this was a good sign that the chunking process was working as intended, which it did.
        tokens_used = response.usage.total_tokens
        df.loc[df['Bill Name'] == bill_number, 'embedded?'] = "yes"
        df.loc[df['Bill Name'] == bill_number, 'tokens_used'] = tokens_used
        print(f"{bill_number} marked as embedded with {tokens_used} tokens used.")
        
        # Save all changes to the CSV after each bill is processed
        df.to_csv(input_csv, index=False)
        print("CSV updated with embedding status and tokens used.")
        
        # adding a delay to avoid rate limiting
        time.sleep(0.1)  

# Run the whole enchilada
generate_and_save_embeddings(df, model, output_folder)

print("Done processing bills over 8,100 tokens.")
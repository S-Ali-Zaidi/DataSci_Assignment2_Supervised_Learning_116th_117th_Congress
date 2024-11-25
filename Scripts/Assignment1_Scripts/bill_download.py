import os
import time
import requests
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent

# Once again, we are loading up our CSV of bills and their URLs linking to the PDFs of the full legislative texts. 
input_csv = 'Data/Main_Data/Bills.csv'
bills_df = pd.read_csv(input_csv).sort_values(by='Bill Name')

# Specifying the folder to save the PDFs in
pdf_folder = 'Data/bill_texts/PDF'
os.makedirs(pdf_folder, exist_ok=True)

# Because extracting over 11,000 PDF's is a timely process, i decided to use a multi-threaded process to speed things up. 
NUM_WORKERS = 3

# I also we getting a lot of errors in downloading all PDF's from Congress.gov, so i used a random User-Agent headers to avoid getting throttled entirely. 
ua = UserAgent()

# this function is to report the number of missing PDFs and existing PDFs in the folder
def report_missing_pdfs():
    total_bills = len(bills_df)
    missing_pdfs = sum(not os.path.exists(os.path.join(pdf_folder, f"{row['Bill Name']}.pdf")) for _, row in bills_df.iterrows())
    existing_pdfs = total_bills - missing_pdfs
    
    print(f"Total bills in CSV: {total_bills}")
    print(f"Bills with existing PDFs: {existing_pdfs}")
    print(f"Bills missing PDFs: {missing_pdfs}")

# and this function is to download the PDFs from the URLs provided in the CSV
# It leverages the use of multi-threading and random User-Agent headers to avoid getting throttled by Congress.gov
# I also use sleep timers to avoid getting throttled by the server.
def download_pdf(bill_name, pdf_url):
    pdf_filename = f'{bill_name}.pdf'
    pdf_filepath = os.path.join(pdf_folder, pdf_filename)

    if os.path.exists(pdf_filepath):
        print(f'{pdf_filename} already exists, skipping download.')
        return

    if pd.notna(pdf_url) and pdf_url.endswith('.pdf'):
        attempt = 0  
        
        while attempt < 5: 
            try:
                
                headers = {'User-Agent': ua.random}
                
                response = requests.get(pdf_url, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(pdf_filepath, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                
                print(f'Successfully downloaded {pdf_filename}')
                # Randomized delay between 0.5 and 5 seconds
                time.sleep(random.uniform(1, 7))
                break  
            
            # here im using a try/except block to catch any exceptions that may occur during the download process
            # exponential backoff is used to slow down my download process if i start hitting rate limits    
            except requests.exceptions.RequestException as e:
                attempt += 1
                print(f'Failed to download {pdf_filename} (Attempt {attempt}): {e}')
                
                if hasattr(response, 'status_code') and response.status_code == 429:
                    wait_time = 2 ** attempt  
                    print(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    break  # Stop retrying if not a 429 error
    else:
        print(f'Invalid or missing PDF URL for {bill_name}')

# This is the main function that will run the entire process of downloading the PDFs
def main():
   
    report_missing_pdfs()
    # Using ThreadPoolExecutor to download PDFs concurrently
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_bill = {
            executor.submit(download_pdf, row['Bill Name'], row['PDF Link']): row['Bill Name']
            for _, row in bills_df.iterrows()
        }
        
        
        for future in as_completed(future_to_bill):
            bill_name = future_to_bill[future]
            try:
                future.result()  
            except Exception as exc:
                print(f'{bill_name} generated an exception: {exc}')

if __name__ == "__main__":
    main()

print('All available PDFs have been downloaded or confirmed as existing.')
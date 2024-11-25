import os
import json
import pandas as pd

# Once again, we are going to parse the JSON files to extract the Short Title and Long Title of each bill, to use for later analysis after clustering. 
# We have already created a csv with bills that match the criteria for my analysis -- having a committee referral listed and a PDF link to download the full text of the bill.
json_input_directory = 'Data/US/2019-2020_116th_Congress/bill'
bills_csv_path = 'Data/Main_Data/Bills.csv'

# Making sure the JSON input directory and Bills CSV file exist
if not os.path.isdir(json_input_directory):
    raise FileNotFoundError(f"JSON input directory not found: {json_input_directory}")

if not os.path.isfile(bills_csv_path):
    raise FileNotFoundError(f"Bills CSV file not found: {bills_csv_path}")

# Loading our Bills CSV into a DataFrame
bills_df = pd.read_csv(bills_csv_path)

# Making sure the 'Bill Name' column exists in the DataFrame, since i had previous iterations of this CSV that used different column names. 
if 'Bill Name' not in bills_df.columns:
    raise KeyError("The input CSV does not contain the 'Bill Name' column.")

# Creating new columns for Short Title and Long Title if they don't already exist
if 'Short Title' not in bills_df.columns:
    bills_df['Short Title'] = ''
if 'Long Title' not in bills_df.columns:
    bills_df['Long Title'] = ''

# creating arrays to store missing JSON files and failed extractions (there were none, but just for peace of mind). 
missing_json_files = []
failed_extractions = []

# Going through each row and attaching .json to the bill name to find the corresponding JSON file.
for index, row in bills_df.iterrows():
    bill_name = row['Bill Name']
    json_filename = f"{bill_name}.json"
    json_filepath = os.path.join(json_input_directory, json_filename)
    
    if not os.path.isfile(json_filepath):
        print(f"JSON file not found for Bill '{bill_name}': {json_filepath}")
        missing_json_files.append(bill_name)
        continue  
    
    # here, we are going to open the JSON file and extract the Short Title and Long Title from the 'bill' section of the JSON file.
    try:
        with open(json_filepath, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        bill_data = data.get('bill', {})
        
        short_title = bill_data.get('title', '').strip()
        long_title = bill_data.get('description', '').strip()
        
        if not short_title:
            print(f"Short Title missing for Bill '{bill_name}' in file {json_filename}")
            failed_extractions.append((bill_name, 'Short Title'))
        
        if not long_title:
            print(f"Long Title missing for Bill '{bill_name}' in file {json_filename}")
            failed_extractions.append((bill_name, 'Long Title'))
        
        # nd then collating this to the DataFrame
        bills_df.at[index, 'Short Title'] = short_title
        bills_df.at[index, 'Long Title'] = long_title
    
    except json.JSONDecodeError:
        print(f"Error decoding JSON for Bill '{bill_name}' in file {json_filename}")
        failed_extractions.append((bill_name, 'JSON Decode Error'))
    except Exception as e:
        print(f"Unexpected error for Bill '{bill_name}' in file {json_filename}: {e}")
        failed_extractions.append((bill_name, str(e)))

# now we merge this new title information back to the CSV file.
bills_df.to_csv(bills_csv_path, index=False)
print(f"\nUpdated Bills CSV with Short and Long Titles saved to: {bills_csv_path}")

# And lastly, let's get a summary of the process and make sure there were not bills that were missed or had failed extractions.
print("\n=== Summary ===")
print(f"Total Bills Processed: {len(bills_df)}")
print(f"Missing JSON Files: {len(missing_json_files)}")
if missing_json_files:
    print("Bills with missing JSON files:")
    for bill in missing_json_files:
        print(f" - {bill}")

print(f"Failed Extractions: {len(failed_extractions)}")
if failed_extractions:
    print("Bills with failed extractions:")
    for bill, reason in failed_extractions:
        print(f" - {bill}: {reason}")

print("\nProcessing complete.")
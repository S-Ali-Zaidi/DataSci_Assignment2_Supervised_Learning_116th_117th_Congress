import os
import json
import csv

# We are pulling data from the Legiscan dataset for all bills in the 117th Congress, stored in JSON format.
# All Relevant data will be saved in a CSV file for further processing.
input_directory = 'Data/US/2019-2020_116th_Congress/bill'
output_directory = 'Data/Main_Data/'

# these are the committees in the House of Representatives that a bill may be referred to upon introduction.
# we need to parse the JSON files to find the committee a bill is referred to 
main_house_committees = {
    "Administration",
    "Agriculture",
    "Appropriations",
    "Armed Services",
    "Budget",
    "Education and Labor",
    "Energy And Commerce",
    "Ethics",
    "Financial Services",
    "Foreign Affairs",
    "Homeland Security",
    "Intelligence",
    "Judiciary",
    "Natural Resources",
    "Oversight and Reform",
    "Rules",
    "Science, Space, And Technology",
    "Small Business",
    "Transportation And Infrastructure",
    "Veterans' Affairs",
    "Ways And Means"
}

# making sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# initialize the output CSV file
output_file = os.path.join(output_directory, 'Bills.csv')

# opening the csv file in write mode
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Bill Name', 'Committee Name', 'PDF Link']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Here we are parsing the json files to extract the committee referall info, as well as any other committee info.
    # This code also parses thejson of each bill to extract the PDF link.
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                bill_name = file.replace('.json', '')  # We will use the fine name for the bill name, as that is how the jsons files are named and structured. 

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        bill = data.get('bill', {})

                        # Variables to store the committee and PDF link
                        committee_found = None
                        pdf_link = None

                        # Here we are checking the 'texts' field in the json file to find the PDF link
                        texts = bill.get('texts', [])
                        if isinstance(texts, list):
                            # If there are multiple text entries, find the first valid PDF link
                            for text_entry in texts:
                                pdf_url = text_entry.get('state_link') or text_entry.get('url')
                                if pdf_url and pdf_url.endswith('.pdf'):
                                    pdf_link = pdf_url
                                    break  

                        # If no valid PDF link, skip to the next bill
                        if not pdf_link:
                            continue

                        # We want to rely on the 'referrals' field to find the committee a bill is referred to, but if that is not available, we will use the 'committee' field.
                        referrals = bill.get('referrals', [])
                        if isinstance(referrals, list):
                            for referral in referrals:
                                if isinstance(referral, dict):  # This is so that we can check the 'chamber' and 'name' fields and ensure we are only pulling Committes from the House of Representatives
                                    chamber = referral.get('chamber', '').upper()
                                    name = referral.get('name', '')
                                    if chamber == 'H' and name in main_house_committees:
                                        committee_found = name
                                        break  # Stop after finding the first valid main committee in referrals

                        # If we didn't find a committee in 'referrals', check the 'committee' field
                        if not committee_found:
                            main_committee = bill.get('committee', {})
                            if isinstance(main_committee, dict):  # Again, we want to make sure the committee listed is in the House of Representatives and in our list of main committees
                                chamber = main_committee.get('chamber', '').upper()
                                name = main_committee.get('name', '')
                                if chamber == 'H' and name in main_house_committees:
                                    committee_found = name

                        # We are filtering the dataset to only use bills that have three necessary components: a valid main committee, a PDF link, and a bill name.
                        if committee_found:
                            writer.writerow({
                                'Bill Name': bill_name,
                                'Committee Name': committee_found,
                                'PDF Link': pdf_link
                            })

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file_path}")
                except Exception as e:
                    print(f"An error occurred while processing file {file_path}: {e}")

print(f"Bills with valid main committees and PDF links have been written to {output_file}")
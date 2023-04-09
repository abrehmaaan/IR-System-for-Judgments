import pandas as pd
import os

# Define the directory where the Excel files are located
directory = './'

# Define the columns that you want to extract
columns_to_extract = ['S.No.', 'Title', 'Case No.', 'Date']

# Create an empty dataframe to store the extracted data
extracted_data = pd.DataFrame(columns=['Category Name', 'S.No.', 'Abstract of Judgments'])

# Loop through each Excel file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Extract the category name from the file name
        category_name = filename.split('.')[0]
        # Read the Excel file into a dataframe
        df = pd.read_excel(os.path.join(directory, filename))
        # Select the columns that you want to extract
        df = df[columns_to_extract]
        # Concatenate the Title, Case No. and Date columns to create the Abstract of Judgments
        df['Abstract of Judgments'] = df['Title'] + ' Case No: ' + df['Case No.'] + ' Date: ' + df['Date'].astype(str)
        # Add the Category Name column to the dataframe
        df['Category Name'] = category_name
        # Append the extracted data to the main dataframe
        extracted_data = pd.concat([extracted_data, df], ignore_index=True)

extracted_data.drop(['Title','Case No.','Date'], axis=1, inplace=True)
# Write the extracted data to a plain text file
extracted_data.to_csv('extracted_data.txt', sep='\t', index=False)

# Create a second dataframe with the Category Name and No of Judgments columns
judgment_counts = extracted_data.groupby('Category Name')['S.No.'].count().reset_index()
judgment_counts.columns = ['Category Name', 'No of Judgments']

# Calculate the total number of judgments
total_judgments = judgment_counts['No of Judgments'].sum()

# Append the total judgments row to the dataframe
total_row = pd.DataFrame({'Category Name': ['Total judgments'], 'No of Judgments': [total_judgments]})
judgment_counts = pd.concat([judgment_counts, total_row], ignore_index=True)

# Write the judgment counts to a second plain text file
judgment_counts.to_csv('judgment_counts.txt', sep='\t', index=False)
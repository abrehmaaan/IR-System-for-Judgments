import os
import pandas as pd

# Set the directory where the Excel files and folders are located
directory = "./"

# Create an empty list to hold the file information
file_info = []

# Loop over each Excel file
for file in os.listdir(directory):
    if file.endswith(".xlsx"):
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel(os.path.join(directory, file))
        # Loop over each row in the DataFrame
        for index, row in df.iterrows():
            # Extract the title and serial number from the row
            title = row["Title"]
            serial_no = str(row["S.No."])
            # Loop over each PDF file in the corresponding folder
            folder_name = os.path.splitext(file)[0]
            folder_path = os.path.join(directory, folder_name)
            for pdf_file in os.listdir(folder_path):
                if title in pdf_file and not pdf_file[0].isdigit():
                    # Rename the PDF file by appending the serial number to the start
                    old_path = os.path.join(folder_path, pdf_file)
                    new_file_name = serial_no + "_" + pdf_file
                    # Check if the filename is too long
                    if len(new_file_name) > 80:
                        new_file_name = new_file_name[:80]
                    if not new_file_name.endswith(".pdf"):
                        new_file_name = new_file_name + ".pdf"
                    new_path = os.path.join(folder_path, new_file_name)
                    os.rename(old_path, new_path)
                    # Add the file information to the list
                    file_info.append([folder_name, serial_no, new_file_name])
                    # Stop looking for PDF files with the same title
                    break

# Write the file information to a plain text file
with open("file_info.txt", "w") as f:
    f.write("Category\tIndex\tDocument Name\n")
    for info in file_info:
        f.write(f"{info[0]}\t{info[1]}\t{info[2]}\n")

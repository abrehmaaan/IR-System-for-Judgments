# define a dictionary to map category to initial
category_to_initial = {
    'administrative': 'A',
    'civil': 'B',
    'commercial': 'C',
    'constitutional': 'D',
    'criminal': 'E',
    'environmental': 'F',
    'family': 'G',
    'tax': 'H'
}

# open the old and new files
with open('file_info.txt', 'r') as old_file, open('newfileinfo.txt', 'w') as new_file:
    # skip the first line since it contains headers
    old_file.readline()
    
    # iterate over the remaining lines in the old file
    for line in old_file:
        # split the line into its components
        category, index, doc_name = line.strip().split('\t')
        
        # generate the new index using the category_to_initial mapping
        initial = category_to_initial.get(category, '')
        doc_id = initial + index
        
        # write the new line to the new file
        new_line = f'{doc_id}\t{doc_name}\n'
        new_file.write(new_line)

# open the old and new files
with open('extracted_data.txt', 'r') as old_file, open('newextracteddata.txt', 'w') as new_file:
    # skip the first line since it contains headers
    old_file.readline()
    
    # iterate over the remaining lines in the old file
    for line in old_file:
        # split the line into its components
        category, index, abstract = line.strip().split('\t')
        
        # generate the new index using the category_to_initial mapping
        initial = category_to_initial.get(category, '')
        doc_id = initial + index
        
        # write the new line to the new file
        new_line = f'{doc_id}\t{abstract}\n'
        new_file.write(new_line)
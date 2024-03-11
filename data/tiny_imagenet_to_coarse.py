import requests
import zipfile
import os
from coarse_fine_tiny_imagenet_labels import *
import shutil

def download_tiny_imagenet(url, path):
    print("Downloading tiny-imagenet-200.zip. Please wait...")
    response = requests.get(url)

    # save the file
    filename = f'{path}/tiny-imagenet-200.zip'

    with open(filename, 'wb') as file:
        file.write(response.content)
        
    print(f"zip file saved to: {filename}")
    
def unzip_imagenet(filename, path):
    print(f"unzipping to: {path}")
    print("please wait...")
    
    # open the zip file in read mode
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        # extract all the contents of the zip file
        zip_ref.extractall()

def filter_train_folder():
    print("Filtering the train folder...")
    
    tiny_imagenet_words = {}

    # open the words.txt file and read the contents
    with open('tiny-imagenet-200/words.txt', 'r') as file:
        # split each line by the tab character
        for line in file:
            line = line.split('\t')
            # add the key value pair to the dictionary
            tiny_imagenet_words[line[0]] = line[1]
            
    with open('tiny-imagenet-200/wnids.txt', 'r') as file:
        wnids = file.read().splitlines()
        
    # create a new dictionary with only the couples in tiny_imagenet_words that have their key in wnids
    tiny_imagenet_words = {key: tiny_imagenet_words[key] for key in wnids}

    # create a new file and write the dictionary to it with tab spacing
    with open('tiny-imagenet-200/words_filtered.txt', 'w') as file:
        for key, value in tiny_imagenet_words.items():
            file.write(f"{key}\t{value}")

    # iterate on all folders of the train set of tiny-imagenet-200
    for folder in os.listdir('tiny-imagenet-200/train'):
        
        if folder not in fine_to_words:
            # remove the folder
            shutil.rmtree(f'tiny-imagenet-200/train/{folder}')
            
    number_of_fine_classes = len(os.listdir('tiny-imagenet-200/train'))
    
    print(f"Number of filtered fine classes: {number_of_fine_classes}")
    
def filter_val_folder():

    val_annotations_path = 'tiny-imagenet-200/val/val_annotations.txt'

    val_to_ids_mapping = {}

    with open(val_annotations_path, 'r') as file:
        # read line by line
        for line in file:
            line = line.split('\t')
            
            key = line[0]
            value = line[1]
            
            if value in fine_to_words:
                
                val_to_ids_mapping[key] = value

    for img in os.listdir('tiny-imagenet-200/val/images'):
        
        if img not in val_to_ids_mapping:
            os.remove(f'tiny-imagenet-200/val/images/{img}')
        else:
            if not os.path.exists(f'tiny-imagenet-200/val/{val_to_ids_mapping[img]}'):
                os.makedirs(f'tiny-imagenet-200/val/{val_to_ids_mapping[img]}')
            
            shutil.move(f'tiny-imagenet-200/val/images/{img}', f'tiny-imagenet-200/val/{val_to_ids_mapping[img]}/{img}')
            
    shutil.rmtree(f'tiny-imagenet-200/val/images/')

def create_a_labeled_test_set():
    print("Creating a labeled test set from the val set...")
    
    shutil.rmtree(f'tiny-imagenet-200/test/')
    if not os.path.exists(f'tiny-imagenet-200/test/'):
        os.makedirs(f'tiny-imagenet-200/test/')
        
    for folder in os.listdir('tiny-imagenet-200/val'):
        if '.txt' not in folder:
            if not os.path.exists(f'tiny-imagenet-200/test/{folder}'):
                os.makedirs(f'tiny-imagenet-200/test/{folder}')
                
            folder_listdir = os.listdir(f'tiny-imagenet-200/val/{folder}')
            
            # Get the number of elements in the list
            num_elements = len(folder_listdir)

            # Define the split point
            split_point = num_elements // 2  # Half for the first part

            # Split the list
            first_part = folder_listdir[:split_point]
            second_part = folder_listdir[split_point:]
            
            # move the second part to the test folder
            for img in second_part:
                shutil.move(f'tiny-imagenet-200/val/{folder}/{img}', f'tiny-imagenet-200/test/{folder}/{img}')

def main():
    
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    path = "."
    
    download_tiny_imagenet(url, path)
    unzip_imagenet('tiny-imagenet-200.zip', path)
    filter_train_folder()
    filter_val_folder()
    create_a_labeled_test_set()
    
    print("Done!")
    
if __name__ == "__main__":
    main()
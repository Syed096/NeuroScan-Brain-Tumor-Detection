import os

path = 'C:\\Users\\TechWizard\\Desktop\\Brain_Tumor_Detection\\Dataset\\augmented_data'

while 'dataset' in path.lower():
    print('Yes, the path contains the word "dataset"')
    print(f"Path: {path}")
    path = os.path.dirname(path)
    
print('No, the path does not contain the word "dataset"')
print(f"Path: {path}")

    

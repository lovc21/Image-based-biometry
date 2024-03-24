import os
from sklearn.model_selection import train_test_split
import shutil
import random

def load_identities(file_path):
    """
    Load identities from the given file and return a dictionary where keys are identities
    and values are lists of image file names.
    """
    identities = {}
    with open(file_path, 'r') as file:
        for line in file:
            image_file, identity = line.strip().split()
            if identity in identities:
                identities[identity].append(image_file)
            else:
                identities[identity] = [image_file]
    return identities

def split_identities(identities, min_images_per_identity=2):
    """
    Split identities into train and test sets, ensuring each identity has at least
    min_images_per_identity in each set.
    """
    train_set = {}
    test_set = {}

    for identity, images in identities.items():
        if len(images) <= 2 * min_images_per_identity:
            random.shuffle(images)
            mid_point = len(images) // 2
            train_set[identity] = images[:mid_point]
            test_set[identity] = images[mid_point:]
        else:
            train, test = train_test_split(images, test_size=0.30, random_state=42)
            train_set[identity] = train
            test_set[identity] = test

    return train_set, test_set

def flatten_identity_sets(identity_sets):
    """
    Flatten the dictionary of identity sets into a list of file names.
    """
    flat_list = []
    for images in identity_sets.values():
        flat_list.extend(images)
    return flat_list

def print_hi(identities_file_path, dataset_path):
    # Load identities and split them
    identities = load_identities(identities_file_path)
    train_identities, test_identities = split_identities(identities)

    # Flatten the sets for easy file handling
    images_train = flatten_identity_sets(train_identities)
    images_test = flatten_identity_sets(test_identities)

    # Modify file paths according to actual image files location
    images_train_full_paths = [os.path.join(dataset_path, f) for f in images_train]
    images_test_full_paths = [os.path.join(dataset_path, f) for f in images_test]

    # Adjust paths for ground truth files
    ground_truths_train = [os.path.join(dataset_path, f.replace('.png', '.txt')) for f in images_train]
    ground_truths_test = [os.path.join(dataset_path, f.replace('.png', '.txt')) for f in images_test]

    # Copy training images and ground truths
    copy_files(images_train_full_paths, 'training/images_ear')
    copy_files(ground_truths_train, 'training/detected_ears')

    # Copy testing images and ground truths
    copy_files(images_test_full_paths, 'test/images_ear')
    copy_files(ground_truths_test, 'test/detected_ears')

def copy_files(file_list, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for file in file_list:
        if os.path.exists(file):
            shutil.copy(file, dest_folder)
        else:
            print(f"Warning: The file {file} does not exist and will not be copied.")

if __name__ == '__main__':
    identities_file = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\identities.txt'
    dataset_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\ears'
    print_hi(identities_file, dataset_path)


import glob
import logging
import os
from skimage.feature import local_binary_pattern, hog
import numpy as np
import cv2
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Normalize, Compose

class Evaluator(object):

    @staticmethod
    def calculate_similarity_matrix(images: list):
        num_images = len(images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(num_images):
                # Calculate cosine similarity between histograms
                first = images[i]
                second = images[j]
                max_length = max(len(first), len(second))
                first = np.pad(first, (0, max_length - len(first)))
                second = np.pad(second, (0, max_length - len(second)))
                if len(first) == 0 or len(second) == 0:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = cosine_similarity(first, second)  # Corrected this line

        return similarity_matrix

    @staticmethod
    def find_most_similar_image(similarity_matrix):
        num_images = similarity_matrix.shape[0]
        most_similar_image = np.zeros(num_images, dtype=int)

        for i in range(num_images):
            # Exclude the image itself from the comparison
            sim_values = np.delete(similarity_matrix[i, :], i)
            most_similar_image[i] = np.argmax(sim_values)

        return most_similar_image

    @staticmethod
    def calculate_accuracy(labels: list, most_similar_image, label_mapping: dict):
        correct_recognitions = 0
        all_recognitions = 0
        for i, similar_image_index in enumerate(most_similar_image):
            query_label = labels[i]
            match_label = labels[similar_image_index]

            all_recognitions += 1
            if query_label == match_label:
                correct_recognitions += 1

        if all_recognitions == 0:
            accuracy = 0
        else:
            accuracy = correct_recognitions / all_recognitions
        return accuracy

def extract_lbp_features(image, P, R, method='default'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P, R, method)
    lbp = lbp.astype("uint64")
    # n_bins = int(lbp.max() + 1)
    n_bins = 256
    hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    # print(hist)
    return hist

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
def load_images_and_labels_separate(dataset_path, identities_file):
    image_files = glob.glob(os.path.join(dataset_path, '*.png'))
    #print(f"Found {len(image_files)} image files in {dataset_path}")
    image_files.sort()
    # Load labels and create a mapping
    label_mapping = {}
    with open(identities_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            # Use the entire filename (including extension) for mapping
            label_mapping[filename] = label
    #print(f"Loaded {len(label_mapping)} labels from {identities_file}")
    images = []
    labels = []
    for file in image_files:
        # Use the entire filename (including extension) for matching
        filename_for_matching = os.path.basename(file).split('-')[1] + '.png'
        #print(filename_for_matching)
        #print(label_mapping)
        if filename_for_matching in label_mapping:
            images.append(cv2.imread(file))
            labels.append(label_mapping[filename_for_matching])
            #print(f"Loaded label {label_mapping[filename_for_matching]} for image file: {file}")
        else:
            print(f"Could not find label for image file: {file}")

    # Debugging: Print the number of images and labels loaded
    print(f"Loaded {len(images)} images and {len(labels)} labels from {dataset_path}")
    #print(f"Labels: {labels}")
    #print(f"Images: {images}")
    return images, labels

def load_resnet50_model(pretrained, num_classes):
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 136)
    # Load the model weights onto CPU
    model.load_state_dict(torch.load('C:\\Users\\Jakob\\PycharmProjects\\SlikovnaBiometrija3\\best_model.pth',
                                     map_location=torch.device('cpu')), strict=False)

    feature_extractor = torch.nn.Sequential(*list(model.children())[:num_classes])
    return feature_extractor
def extract_resnet50_features(images, feature_extractor):
    feature_extractor.eval()  # Set the model to evaluation mode

    # Add normalization as per ResNet50's training
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():  # Disable gradient computation
        for image in images:
            # Convert to PIL image and apply transformations
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            image = image.unsqueeze(0)

            # Extract features using the feature extractor
            feature = feature_extractor(image)
            feature = feature.squeeze().detach().numpy()
            features.append(feature)

    return features

def extract_hog_features(image, orientations, pixels_per_cell,cells_per_block):
    try:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Parameters for HOG can be tweaked for optimization
        features, _ = hog(gray_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, visualize=True)

        return features
    except Exception as e:
        print(
            f"Error processing HOG with orientations={orientations}, pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}: {e}")
        return None


def main():

    trening_dataset_path ='C:\\Users\\Jakob\\Downloads\\yolov8-ears\\datasets\\ears\\images-cropped\\test'
    identities_file_path = 'C:\\Users\\Jakob\\PycharmProjects\\SlikovnaBiometrija3\\identities_test.txt'
    images, labels = load_images_and_labels_separate(trening_dataset_path, identities_file_path)

    # Extract features
    L_list = [8, 16, 24, 32]
    R_list = [1, 2, 3, 4]
    best_accuracy = 0
    for L in L_list:
        for R in R_list:
            lbp_features = [extract_lbp_features(image, P=L, R=R) for image in images]
            evaluator = Evaluator()
            lbp_similarity_matrix = evaluator.calculate_similarity_matrix(lbp_features)
            lbp_most_similar_image = evaluator.find_most_similar_image(lbp_similarity_matrix)
            accuracy_lbp = evaluator.calculate_accuracy(labels, lbp_most_similar_image, labels)
            if accuracy_lbp > best_accuracy:
                best_accuracy = accuracy_lbp
                best_L = L
                best_R = R

    print(f"Best LBP Features Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best L: {best_L}")
    print(f"Best R: {best_R}")
    print("---------------------------------")
    
    
    L_orientations = [8, 16, 24, 32]
    L_pixels_per_cell = [(8, 8), (16, 16), (24, 24), (32, 32)]
    L_cells_per_block = [(1, 1), (2, 2), (3, 3), (4, 4)]
    best_accuracy = 0
    for orientations in L_orientations:
        for pixels_per_cell in L_pixels_per_cell:
            for cells_per_block in L_cells_per_block:
                hog_features = [extract_hog_features(image, orientations, pixels_per_cell,cells_per_block) for image in images]
                evaluator = Evaluator()
                hog_similarity_matrix = evaluator.calculate_similarity_matrix(hog_features)
                hog_most_similar_image = evaluator.find_most_similar_image(hog_similarity_matrix)
                accuracy_hog = evaluator.calculate_accuracy(labels, hog_most_similar_image, labels)
                if accuracy_hog > best_accuracy:
                    best_accuracy = accuracy_hog
                    best_orientations = orientations
                    best_pixels_per_cell = pixels_per_cell
                    best_cells_per_block = cells_per_block
                    print("---------------------------------")
                    print(f"Best HOG Features Accuracy: {best_accuracy * 100:.2f}%")
                    print(f"Best orientations: {best_orientations}")
                    print(f"Best pixels_per_cell: {best_pixels_per_cell}")
                    print(f"Best cells_per_block: {best_cells_per_block}")


    print(f"Best HOG Features Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best orientations: {best_orientations}")
    print(f"Best pixels_per_cell: {best_pixels_per_cell}")
    print(f"Best cells_per_block: {best_cells_per_block}")

    L_pretrained = [True, False]
    L_num_classes = [-1, -2, -3, -4]
    best_accuracy = 0
    for pretrained in L_pretrained:
        for num_classes in L_num_classes:
            feature_extractor = load_resnet50_model(pretrained, num_classes)
            resnet_features = [extract_resnet50_features(image, feature_extractor) for image in images]
            evaluator = Evaluator()
            resnet_similarity_matrix = evaluator.calculate_similarity_matrix(resnet_features)
            resnet_most_similar_image = evaluator.find_most_similar_image(resnet_similarity_matrix)
            accuracy_resnet = evaluator.calculate_accuracy(labels, resnet_most_similar_image, labels)
            if accuracy_resnet > best_accuracy:
                best_accuracy = accuracy_resnet
                best_pretrained = pretrained
                best_num_classes = num_classes
                print("---------------------------------")
                print(f"Best ResNet Features Accuracy: {best_accuracy * 100:.2f}%")
                print(f"Best pretrained: {best_pretrained}")
                print(f"Best num_classes: {best_num_classes}")

    print("---------------------------------")
    print(f"Best ResNet Features Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best pretrained: {best_pretrained}")
    print(f"Best num_classes: {best_num_classes}")

if __name__ == "__main__":
    main()
import math
from skimage.feature import local_binary_pattern
import cv2
import os
import glob
import numpy as np

def extract_lbp_features(image, P, R, method='default'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P, R, method)
    lbp = lbp.astype("uint64")
    #n_bins = int(lbp.max() + 1)
    n_bins = 256
    hist, _ = np.histogram(lbp.flatten(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    #print(hist)
    return hist

def get_pixel(image, x, y):
    try:
        return image[x, y]
    except IndexError:
        return 0

def binary2decimal(binary_list):
    binary_list.reverse()
    ans = 0
    factor = 1
    for i in binary_list:
        ans += i * factor
        factor *= 2
    return ans

def center_compare(center, pixel_list):
    return [1 if pixel >= center else 0 for pixel in pixel_list]

def lbp_manual(image, P, R):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint64)

    for y in range(height):
        for x in range(width):
            center = gray_image[y, x]
            neighbors = []

            for point in range(P):
                # Calculate the position of the neighbor
                rx = x + R * math.cos(2 * math.pi * point / P)
                ry = y - R * math.sin(2 * math.pi * point / P)

                # Bilinear interpolation
                x1, y1 = int(rx), int(ry)
                x2, y2 = math.ceil(rx), math.ceil(ry)

                # Ensure the coordinates are within image boundaries
                x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
                y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

                # Calculate the interpolated pixel value
                fxy1 = gray_image[y1, x1]
                fxy2 = gray_image[y1, x2]
                fxy3 = gray_image[y2, x1]
                fxy4 = gray_image[y2, x2]

                bilinear = fxy1 * (x2 - rx) * (y2 - ry) + fxy2 * (rx - x1) * (y2 - ry) + \
                           fxy3 * (x2 - rx) * (ry - y1) + fxy4 * (rx - x1) * (ry - y1)

                neighbors.append(bilinear)

            binary_list = center_compare(center, neighbors)
            decimal_value = binary2decimal(binary_list)
            lbp_image[y, x] = decimal_value

    # Calculate and normalize the histogram
    n_bins = int(lbp_image.max() + 1)
    n_bins = 256
    hist, _ = np.histogram(lbp_image.flatten(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist
# Function to flatten an image to a vector
def image_to_vector(image, size=(32, 32)):
    resized_image = cv2.resize(image, size)
    return resized_image.flatten()

# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Load images and their corresponding labels
def load_images_and_labels_separate(dataset_path, identities_file):
    image_files = glob.glob(os.path.join(dataset_path, '*.png'))
    print(f"Found {len(image_files)} image files in {dataset_path}")
    image_files.sort()

    # Load labels and create a mapping
    label_mapping = {}
    with open(identities_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            # Use the entire filename (including extension) for mapping
            label_mapping[filename] = label

    images = []
    labels = []
    for file in image_files:
        # Use the entire filename (including extension) for matching
        filename_for_matching = os.path.basename(file)
        if filename_for_matching in label_mapping:
            images.append(cv2.imread(file))
            labels.append(label_mapping[filename_for_matching])
        else:
            print(f"Could not find label for image file: {file}")

    # Debugging: Print the number of images and labels loaded
    print(f"Loaded {len(images)} images and {len(labels)} labels from {dataset_path}")
    # print(f"Labels: {labels}")
    # print(f"Images: {images}")
    return images, labels
def pairwise_comparison(features, labels, threshold_cosine=0.5, threshold_euclidean=0.5):
    n = len(features)
    correct_predictions = 0
    total_comparisons = 0

    max_euclidean_distance = calculate_max_euclidean_distance(features)

    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid comparing the same feature
                total_comparisons += 1

                # Calculate distances
                cosine_sim = cosine_similarity(features[i], features[j])
                euclidean_dist = euclidean_distance(features[i], features[j])

                # Normalize Euclidean distance
                normalized_euclidean_dist = euclidean_dist / max_euclidean_distance

                # Check if the labels match
                is_same_identity = labels[i] == labels[j]

                # Determine if the prediction is correct based on cosine similarity and Euclidean distance
                correct_cosine = (cosine_sim >= threshold_cosine) and is_same_identity
                correct_euclidean = (normalized_euclidean_dist <= threshold_euclidean) and is_same_identity

                if correct_cosine or correct_euclidean:
                    correct_predictions += 1

    accuracy = correct_predictions / total_comparisons
    return accuracy

def comparison_pixle2pixle(vectors, labels):
     number_of_matches = 0
     for i, vector_i in enumerate(vectors):
         min_distance = 1
         best_match_index = -1
         for j, vector_j in enumerate(vectors):
             if i != j:  # Avoid comparing the same vector
                 distance = cosine_distance(vector_i, vector_j)
                 if distance < min_distance:
                     min_distance = distance
                     best_match_index = j

         # Compare identities based on the labels
         if best_match_index != -1:
             if labels[i] == labels[best_match_index]:
                 number_of_matches += 1

     return number_of_matches / len(vectors)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_max_euclidean_distance(features):
    max_dist = 0
    for f1 in features:
        for f2 in features:
            if len(f1) == len(f2):
                dist = euclidean_distance(f1, f2)
                if dist > max_dist:
                    max_dist = dist
            else:
                print(f"Mismatched feature sizes: {len(f1)} and {len(f2)}")
    return max_dist


# Main function
def main():
    trening_dataset_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\outputdata\\detected_ears' #rezane slike
    trening_identities_file = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\outputdata\\identities.txt' # datoteka z imeni slik
    ground_truth_dataset_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\outputdata\\images_ear' #rezane slike ground truth

    train_images, train_labels = load_images_and_labels_separate(trening_dataset_path, trening_identities_file)

    ground_truth_images, ground_truth_labels = load_images_and_labels_separate(ground_truth_dataset_path, trening_identities_file)

    if not train_images or not ground_truth_images:
        print("Training or testing dataset is empty.")
        return

    L_list = [8, 16, 24, 32]
    R_list = [1, 2, 3, 4]

    best_accuracy = 0
    best_params = (0, 0)

    print("LBP Accuracy started training started")
    print("Getting best parameters for LBP")
    for L in L_list:
        for R in R_list:
            #using library LBP
            # Extract LBP features from training images gt
            train_lbp_features = [extract_lbp_features(img, L, R) for img in ground_truth_images]
            #print("library LBP features extracted", train_lbp_features )

            # Pairwise comparison and accuracy calculation
            accuracy = pairwise_comparison(train_lbp_features, ground_truth_labels)
            #print(f"Accuracy for L={L}, R={R}: {accuracy}")

            # Check if these parameters are the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (L, R)

    print("library LBP ended")
    print(f"Best Parameters: L={best_params[0]}, R={best_params[1]} with accuracy {best_accuracy}")

    # Extract LBP features using the library method for test images
    test_lbp_features_library = [extract_lbp_features(img, best_params[0], best_params[1]) for img in train_images]
    test_lbp_features_library_gt = [extract_lbp_features(img, best_params[0], best_params[1]) for img in ground_truth_images]

    # Extract LBP features using the manual method for test images
    test_lbp_features_manual = [lbp_manual(img, best_params[0], best_params[1]) for img in train_images]
    test_lbp_features_manual_gt = [lbp_manual(img, best_params[0], best_params[1]) for img in ground_truth_images]

    # Extract pixel-to-pixel features for test images
    test_pixel_features = [image_to_vector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in train_images]
    test_pixel_features_gt = [image_to_vector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in ground_truth_images]

    # Calculate accuracy for each feature type
    library_lbp_accuracy = pairwise_comparison(test_lbp_features_library, train_labels)
    manual_lbp_accuracy = pairwise_comparison(test_lbp_features_manual, train_labels)

    library_lbp_accuracy_gt = pairwise_comparison(test_lbp_features_library_gt,ground_truth_labels )
    manual_lbp_accuracy_gt = pairwise_comparison(test_lbp_features_manual_gt, ground_truth_labels)

    pixel_accuracy = comparison_pixle2pixle(test_pixel_features, train_labels)
    pixel_accuracy_gt = comparison_pixle2pixle(test_pixel_features_gt, ground_truth_labels)

    print(f"Library LBP Accuracy: {library_lbp_accuracy * 100:.2f}%")
    print(f"Manual LBP Accuracy: {manual_lbp_accuracy * 100:.2f}%")
    print(f"Pixel-to-Pixel Accuracy: {pixel_accuracy * 100:.2f}%")
    print("----------------------------------------")
    print(f"Library LBP Accuracy for GT: {library_lbp_accuracy_gt * 100:.2f}%")
    print(f"Manual LBP Accuracy for GT: {manual_lbp_accuracy_gt * 100:.2f}%")
    print(f"Pixel-to-Pixel Accuracy for GT: {pixel_accuracy_gt * 100:.2f}%")


if __name__ == '__main__':
    main()

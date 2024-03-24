import cv2
import os
import glob
import numpy as np

def crop_and_save_ears(detected_ears, image, image_file, output_path):
    base_name = os.path.basename(image_file)
    base_name_without_extension = os.path.splitext(base_name)[0]
    output_filename = os.path.join(output_path, f"{base_name_without_extension}.png")
    if detected_ears:
        x, y, w, h = detected_ears[0]
        roi = image[y:y + h, x:x + w]
        cv2.imwrite(output_filename, roi)

def crop_and_save_ground_truth(gt_bbox, image, image_file, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    x, y, w, h = gt_bbox
    roi = image[y:y + h, x:x + w]
    base_name = os.path.basename(image_file)
    output_filename = os.path.join(output_path, base_name)

    if roi.size > 0:
        cv2.imwrite(output_filename, roi)
    else:
        print(f"ROI is empty for file {image_file}")
def read_ground_truth(ground_truth_file, image_width, image_height):
    try:
        with open(ground_truth_file, 'r') as file:
            data = file.read().strip().split()
        x_center_norm, y_center_norm, width_norm, height_norm = map(float, data[1:])
        x_center = x_center_norm * image_width
        y_center = y_center_norm * image_height
        width = width_norm * image_width
        height = height_norm * image_height
        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        return (x1, y1, int(width), int(height))
    except Exception as e:
        print(f"Error reading ground truth file {ground_truth_file}: {e}")
        return None

def calculate_iou(boxA, boxB):
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def grid_search_parameters(image, gray_image, cascade_classifier, ground_truth_bbox):
    scale_factors = np.arange(1.01, 1.4, 0.01)
    min_neighbors = range(3, 10)
    min_sizes = [(30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100)]
    best_iou = 0
    best_params = None
    for scale_factor in scale_factors:
        for min_neighbor in min_neighbors:
            for min_size in min_sizes:
                detected_ears = cascade_classifier.detectMultiScale(
                    gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbor, minSize=min_size
                )
                for ear_bbox in detected_ears:
                    iou = calculate_iou(ground_truth_bbox, ear_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_params = (scale_factor, min_neighbor, min_size)
    print(f"Best IoU: {best_iou} Best parameters: {best_params}")
    return best_params, best_iou

def detect_ears():
    dataset_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\training\\images_ear'
    output_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\outputdata\\detected_ears'
    ground_truth_file_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\training\\detected_ears'
    output_path_ear_gt = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\outputdata\\images_ear'
    test_path = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\test\\images_ear'
    ground_truth_file_test = 'C:\\Users\\Jakob\\PycharmProjects\\task1SlikovnaBiometrija\\test\\detected_ears'

    image_files = glob.glob(os.path.join(dataset_path, '*.png'))
    test_image_files = glob.glob(os.path.join(test_path, '*.png'))

    cascade_file_leftear = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
    cascade_file_rightear = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

    ious_for_params = {}
    # run grid search for each training image and store the best parameters
    for image_file in image_files:
        image = cv2.imread(image_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_name = os.path.basename(image_file).split('.')[0]
        ground_truth_file = os.path.join(ground_truth_file_path, f"{image_name}.txt")

        if not os.path.exists(ground_truth_file):
            print(f"Ground truth file does not exist for {image_name}, skipping.")
            continue

        gt_bbox = read_ground_truth(ground_truth_file, image.shape[1], image.shape[0])

        if gt_bbox:
            crop_and_save_ground_truth(gt_bbox, image, image_file, output_path_ear_gt)

        if gt_bbox is None:
            print(f"No ground truth data for {image_name}, skipping.")
            continue

        best_params, best_iou = grid_search_parameters(image, gray_image, cascade_file_leftear, gt_bbox)
        best_params_right, best_iou_right = grid_search_parameters(image, gray_image, cascade_file_rightear, gt_bbox)

        ious_for_params.setdefault(best_params, []).append(best_iou)
        ious_for_params.setdefault(best_params_right, []).append(best_iou_right)


    print(f"Best parameters for ears: {ious_for_params}")
    
    # Variables to store the sum of IoUs and parameter occurrences
    total_iou = 0
    param_occurrences = {}
    total_images = 0

    # Iterate over the ious_for_params dictionary
    for params, ious in ious_for_params.items():
        if params not in param_occurrences:
            param_occurrences[params] = 0

        total_iou += sum(ious)
        param_occurrences[params] += len(ious)
        total_images += len(ious)

    # Calculate the average IoU across all images
    average_iou = total_iou / total_images if total_images > 0 else 0

    # Find the most frequently used parameters
    most_common_params = max(param_occurrences, key=param_occurrences.get)

    print(f"Average IoU across all images: {average_iou}")
    print(f"Most common parameters: {most_common_params}")

    """
    print(f"The mean IOU is {mean_iou}")
    print(f"Overall best IoU: {overall_best_iou}")
    print(f"Overall best parameters: {overall_best_params}")
    """

    with open('results3.txt', 'w') as file:
        file.write(f"Overall best IoU: {average_iou}\n")
        file.write(f"Overall best parameters: {most_common_params}\n")

    # Detect ears in test images using the most common parameters
    for image_file in test_image_files:
        image = cv2.imread(image_file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        left_ears = cascade_file_leftear.detectMultiScale(
            gray_image, scaleFactor=most_common_params[0], minNeighbors=most_common_params[1],
            minSize=most_common_params[2]
        )
        right_ears = cascade_file_rightear.detectMultiScale(
            gray_image, scaleFactor=most_common_params[0], minNeighbors=most_common_params[1],
            minSize=most_common_params[2]
        )

        detected_ears = list(left_ears) + list(right_ears)
        crop_and_save_ears(detected_ears, image, image_file, output_path)

        image_name = os.path.basename(image_file).split('.')[0]
        ground_truth_file = os.path.join(ground_truth_file_test, f"{image_name}.txt")

        if os.path.exists(ground_truth_file):
            gt_bbox = read_ground_truth(ground_truth_file, image.shape[1], image.shape[0])
            if gt_bbox:
                crop_and_save_ground_truth(gt_bbox, image, image_file, output_path_ear_gt)

        print(f"Detected {len(detected_ears)} ears in {image_file}")

if __name__ == '__main__':
    detect_ears()
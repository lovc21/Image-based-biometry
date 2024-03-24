import os
from PIL import Image

err_list = ['564-22']
translation_map = {}

with open('identities_test.txt', 'r') as file:
    test_ids = {line.split()[0].replace('.png', ''): line.split()[1] for line in file}


def parse_yolo_annotation(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        _, x_center, y_center, width, height = map(float, line.split())
        boxes.append((x_center, y_center, width, height))
    return boxes

def convert_to_pixel_coordinates(box, img_width, img_height):
    x_center, y_center, width, height = box
    x = int((x_center - width / 2) * img_width)
    y = int((y_center - height / 2) * img_height)
    w = int(width * img_width)
    h = int(height * img_height)
    return x, y, w, h

def cut_and_save_image(image_path, annotation_path, save_path, id, img_name):
    image = Image.open(image_path)
    img_width, img_height = image.size
    boxes = parse_yolo_annotation(annotation_path)

    for i, box in enumerate(boxes):
        x, y, w, h = convert_to_pixel_coordinates(box, img_width, img_height)
        cut_image = image.crop((x, y, x+w, y+h))
        cut_image.save(os.path.join(save_path, f'{id}-{img_name}-{i}.png'))

def process_images(base_path, categories, out_dir):
    global translation_map
    for category in categories:
        progress = 0
        progress_prev = -1

        img_dir = os.path.join(base_path, 'images', category)
        label_dir = os.path.join(base_path, 'labels', category)
        save_dir = os.path.join(out_dir, category)
        os.makedirs(save_dir, exist_ok=True)

        image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        total_images = len(image_files)

        id_list = []
        for ii in image_files:
            id_list.append(os.path.basename(ii).split('-')[0])
        uniq_ids = set(id_list)
        translation_map = {value: index for index, value in enumerate(uniq_ids)}
        num_classes = len(uniq_ids) # Get the number of unique classes

        for i, filename in enumerate(image_files):
            if os.path.splitext(os.path.basename(filename))[0] in err_list:
                continue
            image_path = os.path.join(img_dir, filename)
            annotation_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
            img_split_name = os.path.splitext(os.path.basename(image_path))[0].split('-')
            
            if category == 'test':
                id = int(test_ids[img_split_name[0]])
                img_name = img_split_name[0]
            else:
                id = translation_map[img_split_name[0]]
                img_name = img_split_name[1]

            if category == 'val':
                id += 1000 # Add 1000 to avoid overlap with train
            elif category == 'test':
                id += 2000 # Add 2000 to avoid overlap with train and val

            cut_and_save_image(image_path, annotation_path, save_dir, id, img_name)

            progress = round((i + 1) / total_images * 100)
            if progress != progress_prev:
                print(f"Processed {progress:d}% of {category} images.", end='\r', flush=True)
            progress_prev = progress
        print(" " * 50, end='\r')           
        print(f"Processed all {category} images.")

base_path = os.path.join('C:\\Users\\Jakob\\Downloads\\yolov8-ears\\datasets', 'ears')  # Replace with your base directory
out_dir = os.path.join(base_path, 'images-cropped')
categories = ['test', 'train', 'val']
process_images(base_path, categories, out_dir)
print(f"Cropped images stored in {out_dir}.")

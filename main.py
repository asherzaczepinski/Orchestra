#WHEN I HAVE A LOT OF IMAGES IT OVERRIDES IT I HAVE TO FIX THIS


from preprocessing import *
from staff_removal import *
from helper_methods import *

import argparse
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import json

# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input Folder", default="input", nargs='?')
parser.add_argument("outputfolder", help="Output Folder", default="output", nargs='?')

# Parse the arguments
args = parser.parse_args()

# Threshold for line to be considered as an initial staff line #
threshold = 0.8
filename = 'model/model.sav'
model = pickle.load(open(filename, 'rb'))
accidentals = ['x', 'hash', 'b', 'symbol_bb', 'd']
bounding_boxes_dict = {}
global_counter = 0  # Global counter for image naming

def draw_bounding_boxes(image, boundaries):
    """
    Draws bounding boxes on the image without labels.
    
    :param image: The image on which to draw.
    :param boundaries: List of boundary tuples (x1, y1, x2, y2).
    """
    for boundary in boundaries:
        x1, y1, x2, y2 = boundary
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
def save_bounding_box_areas(image, boundaries, labels, outputfolder, base_filename):
    global global_counter  # Use the global counter

    print(f"Starting to save bounding box areas and context images in {outputfolder}...")

    if not boundaries:
        print("No boundaries provided to save_bounding_box_areas.")
        return

    for boundary in boundaries:
        print(f"Processing boundary with global counter {global_counter}: {boundary}")
        x1, y1, x2, y2 = boundary
        cropped_image = image[y1:y2, x1:x2]

        # Check for valid crop
        if cropped_image.size == 0:
            print(f"Boundary resulted in an empty crop. Skipping.")
            continue

        # Save the cropped image
        bbox_filename = f"smaller_image_{global_counter}.png"
        bbox_filepath = os.path.join(outputfolder, bbox_filename)
        print(f"Saving cropped image at {bbox_filepath}")
        cv2.imwrite(bbox_filepath, cropped_image)

        # Create a copy of the original image for context
        image_with_context = image.copy()

        # Highlight the specific bounding box in the context image
        cv2.rectangle(image_with_context, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save the context image
        context_filename = f"smaller_image_w_context_{global_counter}.png"
        context_filepath = os.path.join(outputfolder, context_filename)
        print(f"Saving context image at {context_filepath}")
        cv2.imwrite(context_filepath, image_with_context)

        global_counter += 1  # Increment the global counter

    print("Finished saving bounding box areas and context images.")

def process_and_display_image(inputfolder, fn, outputfolder):
    cutted, clean_image = preprocessing(inputfolder, fn, None)
    height_before = 0
    all_boundaries = []
    all_boundaries_offset_adjusted = []

    for it in range(len(cutted)):
        symbols_boundaries = segmentation(height_before, cutted[it])
        symbols_boundaries.sort(key=lambda x: (x[0], x[1]))

        for boundary in symbols_boundaries:
            [[_, cutted_boundaries], x1, y1, x2, y2] = get_label_cutted_boundaries(boundary, height_before, cutted[it])
            all_boundaries.extend(cutted_boundaries)
            all_boundaries_offset_adjusted.append((x1, y1, x2, y2))

        height_before += cutted[it].shape[0]

    # Draw bounding boxes on the cleaned image without labels
    image_with_boxes = clean_image.copy()
    draw_bounding_boxes(image_with_boxes, all_boundaries_offset_adjusted)

    # Save the annotated image with line removal in place
    output_image_filepath = os.path.join(outputfolder, f'annotated_{fn}')
    cv2.imwrite(output_image_filepath, image_with_boxes)

    print(f"Annotated image saved at {output_image_filepath}")

    # Save bounding box areas as separate images
    base_filename = os.path.splitext(fn)[0]
    save_bounding_box_areas(clean_image, all_boundaries_offset_adjusted, [], outputfolder, base_filename)

def preprocessing(inputfolder, fn, f):
      # Get image and its dimensions #
    height, width, in_img = preprocess_img('{}/{}'.format(inputfolder, fn))
    
    # Get line thinkness and list of staff lines #
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, in_img, threshold)

    # Remove staff lines from original image #
    cleaned = remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses)
    
    # Get list of cutted buckets and cutting positions #
    cut_positions, cutted = cut_image_into_buckets(cleaned, staff_lines)
    
    print(cutted)
    # Get reference line for each bucket #

    return cutted, cleaned

def get_target_boundaries(label, cur_symbol, y2):
    if label == 'b_8':
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = 'a_8'
    elif label == 'b_8_flipped':
        cutted_boundaries = cut_boundaries(cur_symbol, 2, y2)
        label = 'a_8_flipped'
    elif label == 'b_16':
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = 'a_16'
    elif label == 'b_16_flipped':
        cutted_boundaries = cut_boundaries(cur_symbol, 4, y2)
        label = 'a_16_flipped'
    else: 
        cutted_boundaries = cut_boundaries(cur_symbol, 1, y2)

    return label, cutted_boundaries

def get_label_cutted_boundaries(boundary, height_before, cutted):
    # Get the current symbol #
    print("Original boundary:", boundary) 

    x1, y1, x2, y2 = boundary

    cur_symbol = cutted[y1-height_before:y2+1-height_before, x1:x2+1]

    # Clean and cut #
    cur_symbol = clean_and_cut(cur_symbol)
    cur_symbol = 255 - cur_symbol

    # Start prediction of the current symbol #
    feature = extract_hog_features(cur_symbol)
    label = str(model.predict([feature])[0])
    return [get_target_boundaries(label, cur_symbol, y2), x1, y1, x2, y2]

def main():
    saved_images_folder = os.path.join(args.outputfolder, 'saved_images')
    if not os.path.exists(saved_images_folder):
        os.makedirs(saved_images_folder)
    if not os.path.exists(args.outputfolder):
        try:
            os.mkdir(args.outputfolder)
        except OSError as error:
            print("Error creating output folder:", error)
            return
    else:
        print("Output folder already exists. Using existing folder.")

    list_of_images = os.listdir(args.inputfolder)
    for fn in list_of_images:
        # Check if the file is an image
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        full_path = os.path.join(args.inputfolder, fn)
        if not os.path.isfile(full_path):
            print("File does not exist:", fn)
            continue

        # Process and display each image
        try:
            process_and_display_image(args.inputfolder, fn, saved_images_folder)
        except Exception as e:
            print(e)
            print(f'Processing failed for {fn}')

    print('Processing completed for all images.') 

if __name__ == "__main__":
    main()



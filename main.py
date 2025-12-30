#WHEN I HAVE A LOT OF IMAGES IT OVERRIDES IT I HAVE TO FIX THIS

from classifier import *
from preprocessing import *
from staff_removal import *
from helper_methods import *

import argparse
import os
import datetime
import pickle
import cv2
import numpy as np

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input Folder", default="input", nargs='?')
parser.add_argument("outputfolder", help="Output Folder", default="output", nargs='?')

args = parser.parse_args()

# Threshold for line to be considered as an initial staff line #
threshold = 0.8
accidentals = ['x', 'hash', 'b', 'symbol_bb', 'd']

filename = 'model/model.sav'
model = pickle.load(open(filename, 'rb'))

def draw_continuous_segments(img, x, top, bottom, cleaned_img, color, thickness):
    """Helper function to draw only continuous black segments of a vertical line"""
    # Collect all non-white pixels in this vertical range (threshold: < 250)
    black_y = []
    for y in range(top, bottom + 1):
        if y < cleaned_img.shape[0] and cleaned_img[y, x] < 250:
            black_y.append(y)

    if len(black_y) == 0:
        return

    # Draw continuous segments
    seg_start = black_y[0]
    for i in range(1, len(black_y)):
        if black_y[i] > black_y[i-1] + 1:  # Gap detected
            cv2.line(img, (x, seg_start), (x, black_y[i-1]), color, thickness)
            seg_start = black_y[i]
    # Draw final segment
    cv2.line(img, (x, seg_start), (x, black_y[-1]), color, thickness)

def detect_measure_lines(cleaned_img, staff_lines, width, height, outputfolder, file_prefix):
    """
    Detect measure lines (barlines) in the cleaned image.
    Returns list of x-coordinates where valid barlines are found.
    Saves debug images at each step.
    """
    # Create debug folder
    debug_folder = f"{outputfolder}/debug_{file_prefix}"
    os.makedirs(debug_folder, exist_ok=True)

    # Create a color version of the image for visualization
    vis_base = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)

    # Calculate histogram of vertical non-white pixels (< 250)
    col_histogram = np.sum(cleaned_img < 250, axis=0)

    # Find candidate columns with enough black pixels
    threshold_density = height * 0.1  # At least 10% black pixels
    candidate_columns = []
    for c in range(width):
        if col_histogram[c] > threshold_density:
            candidate_columns.append(c)

    # STEP 1: Show all candidate columns (only where non-white pixels exist)
    step1_img = vis_base.copy()
    for c in candidate_columns:
        # Find continuous segments of non-white pixels in this column
        in_segment = False
        seg_start = None
        for y in range(height):
            if cleaned_img[y, c] < 250:  # Non-white pixel
                if not in_segment:
                    seg_start = y
                    in_segment = True
            else:  # White pixel
                if in_segment:
                    # Draw the segment we just finished
                    cv2.line(step1_img, (c, seg_start), (c, y-1), (255, 0, 0), 1)
                    in_segment = False
        # Draw final segment if we ended on non-white
        if in_segment:
            cv2.line(step1_img, (c, seg_start), (c, height-1), (255, 0, 0), 1)
    cv2.imwrite(f"{debug_folder}/step1_all_candidates.png", step1_img)
    print(f"  Step 1: Found {len(candidate_columns)} candidate columns")

    # Group nearby candidates into line segments
    line_segments = []
    current_segment = []
    for i, col in enumerate(candidate_columns):
        if len(current_segment) == 0:
            current_segment.append(col)
        elif col - current_segment[-1] <= 3:  # Within 3 pixels
            current_segment.append(col)
        else:
            line_segments.append(current_segment)
            current_segment = [col]
    if len(current_segment) > 0:
        line_segments.append(current_segment)

    # Calculate extents for each segment and prepare visualization
    step2_img = vis_base.copy()
    line_segments_extents = []
    for segment in line_segments:
        line_x = int(np.mean(segment))
        # Find all continuous non-white segments for this line
        black_regions = set()  # Store all y-coordinates that have non-white pixels
        for x in segment:
            for y in range(height):
                if cleaned_img[y, x] < 250:
                    black_regions.add(y)

        if len(black_regions) > 0:
            seg_top = min(black_regions)
            seg_bottom = max(black_regions)
            line_segments_extents.append((line_x, seg_top, seg_bottom))

            # Draw only continuous black segments
            sorted_y = sorted(black_regions)
            seg_start = sorted_y[0]
            for i in range(1, len(sorted_y)):
                if sorted_y[i] > sorted_y[i-1] + 1:  # Gap detected
                    cv2.line(step2_img, (line_x, seg_start), (line_x, sorted_y[i-1]), (0, 255, 255), 2)
                    seg_start = sorted_y[i]
            # Draw final segment
            cv2.line(step2_img, (line_x, seg_start), (line_x, sorted_y[-1]), (0, 255, 255), 2)

    cv2.imwrite(f"{debug_folder}/step2_grouped_segments.png", step2_img)
    print(f"  Step 2: Grouped into {len(line_segments)} line segments")

    # Filter by width (barlines should be relatively thin)
    max_barline_width = 15
    after_width_filter = []
    for x, top, bottom in line_segments_extents:
        # Check width at multiple heights
        widths = []
        for y in range(top, bottom, max(1, (bottom - top) // 5)):
            if y >= height:
                continue
            # Count consecutive non-white pixels horizontally
            width_count = 0
            for dx in range(-max_barline_width, max_barline_width):
                if 0 <= x + dx < width and cleaned_img[y, x + dx] < 250:
                    width_count += 1
                else:
                    break
            widths.append(width_count)

        if len(widths) > 0:
            avg_width = np.mean(widths)
            if avg_width <= max_barline_width:
                after_width_filter.append((x, top, bottom))

    # STEP 3: Show after width filter
    step3_img = vis_base.copy()
    for x, top, bottom in after_width_filter:
        draw_continuous_segments(step3_img, x, top, bottom, cleaned_img, (255, 255, 0), 2)
    cv2.imwrite(f"{debug_folder}/step3_after_width_filter.png", step3_img)
    print(f"  Step 3: {len(after_width_filter)} lines after width filter")

    # Filter by height (barlines should span significant vertical distance)
    if staff_lines is not None and len(staff_lines) > 0:
        # Calculate typical staff spacing
        staff_heights = []
        for staff in staff_lines:
            if isinstance(staff, (list, tuple)) and len(staff) >= 2:
                staff_heights.append(staff[-1] - staff[0])
        min_line_height = np.mean(staff_heights) * 0.6 if len(staff_heights) > 0 else height * 0.15
    else:
        min_line_height = height * 0.15

    after_height_filter = []
    for x, top, bottom in after_width_filter:
        line_height = bottom - top
        if line_height >= min_line_height:
            after_height_filter.append((x, top, bottom, line_height))

    # STEP 4: Show after height filter
    step4_img = vis_base.copy()
    for x, top, bottom, h in after_height_filter:
        draw_continuous_segments(step4_img, x, top, bottom, cleaned_img, (128, 0, 128), 2)
        cv2.putText(step4_img, f"{h:.0f}", (x+5, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f"{debug_folder}/step4_after_height_filter.png", step4_img)
    print(f"  Step 4: {len(after_height_filter)} lines after height filter (min: {min_line_height:.1f})")

    # Filter by alignment with staff lines
    after_alignment_filter = []
    for x, top, bottom, _ in after_height_filter:
        # Check if line intersects with staff lines
        intersects_staff = False
        if staff_lines is not None:
            for staff in staff_lines:
                if isinstance(staff, (list, tuple)):
                    for staff_y in staff:
                        if top <= staff_y <= bottom:
                            intersects_staff = True
                            break
                    if intersects_staff:
                        break

        # Check if it's a single barline or double barline
        is_single = True
        for x2, top2, bottom2, _ in after_height_filter:
            if x != x2 and abs(x - x2) < 20:  # Close to another line
                is_single = False
                break

        if intersects_staff:
            after_alignment_filter.append((x, top, bottom, is_single))

    # STEP 5: Show after alignment filter
    step5_img = vis_base.copy()
    for x, top, bottom, is_single in after_alignment_filter:
        color = (0, 128, 255) if is_single else (255, 128, 0)
        draw_continuous_segments(step5_img, x, top, bottom, cleaned_img, color, 2)
    cv2.imwrite(f"{debug_folder}/step5_after_alignment_filter.png", step5_img)
    print(f"  Step 5: {len(after_alignment_filter)} lines after alignment filter")

    # Filter by straightness (barlines should be relatively straight)
    after_curve_filter = []
    for x, top, bottom, _ in after_alignment_filter:
        # Check if the line deviates too much horizontally
        x_positions = []
        for y in range(top, bottom):
            if y >= height:
                continue
            # Find x position of non-white pixel near this y
            found = False
            for dx in range(-5, 6):
                if 0 <= x + dx < width and cleaned_img[y, x + dx] < 250:
                    x_positions.append(x + dx)
                    found = True
                    break
            if not found:
                x_positions.append(x)

        if len(x_positions) > 0:
            x_std = np.std(x_positions)
            if x_std < 5:  # Not too curvy
                after_curve_filter.append((x, top, bottom))

    # STEP 6: Show after curve filter
    step6_img = vis_base.copy()
    for x, top, bottom in after_curve_filter:
        draw_continuous_segments(step6_img, x, top, bottom, cleaned_img, (0, 255, 128), 2)
    cv2.imwrite(f"{debug_folder}/step6_after_curve_filter.png", step6_img)
    print(f"  Step 6: {len(after_curve_filter)} lines after curve filter")

    # Final valid barlines
    valid_barlines = after_curve_filter

    # STEP 7: Show final valid barlines
    step7_img = vis_base.copy()
    for x, top, bottom in valid_barlines:
        draw_continuous_segments(step7_img, x, top, bottom, cleaned_img, (0, 0, 255), 3)
    cv2.imwrite(f"{debug_folder}/step7_final_valid_barlines.png", step7_img)
    print(f"  Step 7: {len(valid_barlines)} FINAL valid barlines")

    # Return just the x-coordinates for the measure visualization
    return sorted([x for x, _, _ in valid_barlines])

def create_measures_visualization(cleaned_img, measure_lines, staff_lines):
    """
    Create a visualization showing detected measures (regions between barlines).
    """
    # Convert to color
    vis_img = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)
    height = cleaned_img.shape[0]

    # Draw staff lines in light gray
    if staff_lines is not None:
        for staff in staff_lines:
            if isinstance(staff, (list, tuple)):
                for y in staff:
                    cv2.line(vis_img, (0, y), (vis_img.shape[1], y), (200, 200, 200), 1)

    # Draw barlines in red
    for x in measure_lines:
        cv2.line(vis_img, (x, 0), (x, height), (0, 0, 255), 2)

    # Highlight measure regions with alternating colors
    colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255)]
    for i in range(len(measure_lines) - 1):
        x1 = measure_lines[i]
        x2 = measure_lines[i + 1]
        color = colors[i % len(colors)]

        # Draw semi-transparent overlay
        overlay = vis_img.copy()
        cv2.rectangle(overlay, (x1, 0), (x2, height), color, -1)
        cv2.addWeighted(overlay, 0.2, vis_img, 0.8, 0, vis_img)

        # Add measure number
        measure_num = i + 1
        text_x = (x1 + x2) // 2
        cv2.putText(vis_img, f"M{measure_num}", (text_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return vis_img

def preprocessing(inputfolder, fn, f, outputfolder):
    # Get image and its dimensions #
    height, width, in_img = preprocess_img('{}/{}'.format(inputfolder, fn))

    # Get line thinkness and list of staff lines #
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, in_img, threshold)

    # Remove staff lines from original image #
    cleaned = remove_staff_lines(in_img, width, staff_lines, staff_lines_thicknesses)

    # Save the cleaned image (staff lines removed) #
    file_prefix = fn.split('.')[0]
    cv2.imwrite(f"{outputfolder}/{file_prefix}_cleaned.png", cleaned)

    # Detect measure lines in the cleaned image #
    print(f"Processing {fn}...")
    measure_lines = detect_measure_lines(cleaned, staff_lines, width, height, outputfolder, file_prefix)

    # Create and save visualization with measures highlighted #
    measures_visualization = create_measures_visualization(cleaned, measure_lines, staff_lines)
    cv2.imwrite(f"{outputfolder}/{file_prefix}_measures.png", measures_visualization)
    print(f"  Saved measures visualization with {len(measure_lines)} barlines")

    # Get list of cutted buckets and cutting positions #
    cut_positions, cutted = cut_image_into_buckets(cleaned, staff_lines)

    # Get reference line for each bucket #
    ref_lines, lines_spacing = get_ref_lines(cut_positions, staff_lines)

    return cutted, ref_lines, lines_spacing

def process_image(inputfolder, fn, f, outputfolder):
    cutted, ref_lines, lines_spacing = preprocessing(inputfolder, fn, f, outputfolder)

    last_acc = ''
    last_num = ''
    height_before = 0

    if len(cutted) > 1:
        f.write('{\n')

    for it in range(len(cutted)):
        f.write('[')
        is_started = False
        cur_img = cutted[it].copy()

        symbols_boundries = segmentation(height_before, cutted[it])
        symbols_boundries.sort(key = lambda x: (x[0], x[1]))

        symbols = []
        for boundry in symbols_boundries:
            # Get the current symbol #
            x1, y1, x2, y2 = boundry
            cur_symbol = cutted[it][y1-height_before:y2+1-height_before, x1:x2+1]

            # Clean and cut #
            cur_symbol = clean_and_cut(cur_symbol)
            cur_symbol = 255 - cur_symbol

            # Start prediction of the current symbol #
            feature = extract_features(cur_symbol, 'hog')
            label = str(model.predict([feature])[0])

            if label == 'clef':
                is_started = True

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

            for cutted_boundary in cutted_boundaries:
                _, y1, _, y2 = cutted_boundary
                if is_started == True and label != 'barline' and label != 'clef':
                    text = text_operation(label, ref_lines[it], lines_spacing[it], y1, y2)

                    if (label == 't_2' or label == 't_4') and last_num == '':
                        last_num = text
                    elif label in accidentals:
                        last_acc = text
                    else:
                        if last_acc != '':
                            text = text[0] + last_acc + text[1:]
                            last_acc = ''

                        if last_num != '':
                            text = f'\meter<"{text}/{last_num}">'
                            last_num = ''

                        not_dot = label != 'dot'
                        f.write(not_dot * ' ' + text)

        height_before += cutted[it].shape[0]
        f.write(' ]\n')

    if len(cutted) > 1:
        f.write('}')

def main():
    # Create output folder if it doesn't exist
    try:
        os.makedirs(args.outputfolder, exist_ok=True)
    except OSError as error:
        print("Error creating output folder:", error)
        return

    # Write metadata file
    with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
        text_file.write("Input Folder: %s\n" % args.inputfolder)
        text_file.write("Output Folder: %s\n" % args.outputfolder)
        text_file.write("Date: %s\n" % datetime.datetime.now())

    list_of_images = os.listdir(args.inputfolder)

    for i, fn in enumerate(list_of_images):
        # Check if the file is an image
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        full_path = os.path.join(args.inputfolder, fn)
        if not os.path.isfile(full_path):
            print("File does not exist:", fn)
            continue

        # Open the output text file #
        file_prefix = fn.split('.')[0]
        f = open(f"{args.outputfolder}/{file_prefix}.txt", "w")

        # Process each image separately #
        try:
            process_image(args.inputfolder, fn, f, args.outputfolder)
            print(f"Successfully processed {fn}")
        except Exception as e:
            print(f'{args.inputfolder}/{fn} has been failed: {e}')

        f.close()

    print('Finished !!')

if __name__ == "__main__":
    main()

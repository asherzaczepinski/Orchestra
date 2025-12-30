from preprocessing import *
from staff_removal import *
from helper_methods import *

import argparse
import os
import datetime
import cv2
import numpy as np
import shutil

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", nargs='?', default="input", help = "Input File")
parser.add_argument("outputfolder", nargs='?', default="output", help = "Output File")

args = parser.parse_args()

with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
    text_file.write("Input Folder: %s" % args.inputfolder)
    text_file.write("Output Folder: %s" % args.outputfolder)
    text_file.write("Date: %s" % datetime.datetime.now())


# Threshold for line to be considered as an initial staff line #
threshold = 0.8
filename = 'model/model.sav'
model = pickle.load(open(filename, 'rb'))
accidentals = ['x', 'hash', 'b', 'symbol_bb', 'd']

def draw_continuous_segments(img, x, top, bottom, cleaned_img, color, thickness):
    """Helper function to draw only continuous black segments of a vertical line"""
    # Collect all black pixels in this vertical range
    black_y = []
    for y in range(top, bottom + 1):
        if y < cleaned_img.shape[0] and cleaned_img[y, x] == 0:
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

    # Convert to color for visualizations
    vis_base = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)

    # Calculate staff heights (from first line to fifth line of each staff)
    no_of_staves = len(staff_lines) // 5
    staff_heights = []
    staff_positions = []  # (top, bottom) for each staff

    for i in range(no_of_staves):
        top_line = staff_lines[i * 5]
        bottom_line = staff_lines[i * 5 + 4]
        staff_height = bottom_line - top_line
        staff_heights.append(staff_height)
        staff_positions.append((top_line, bottom_line))

    if len(staff_heights) == 0:
        return []

    avg_staff_height = np.mean(staff_heights)
    min_line_height = avg_staff_height * 0.95  # Within 5% of staff height
    max_single_staff_height = avg_staff_height * 1.05
    max_double_staff_height = avg_staff_height * 2.1  # For double barlines spanning 2 staves

    print(f"  Staff detection: {no_of_staves} staves, avg height: {avg_staff_height:.1f} pixels")

    # Create column histogram to detect vertical lines
    col_histogram = np.zeros(width)
    for c in range(width):
        col_histogram[c] = np.sum(cleaned_img[:, c] == 0)  # Count black pixels

    # Find candidate vertical lines (columns with high black pixel density)
    threshold_density = height * 0.05  # Lowered to 5% to catch more candidates
    candidate_columns = []

    for c in range(width):
        if col_histogram[c] > threshold_density:
            candidate_columns.append(c)

    # STEP 1: Show all candidate columns (only where black pixels exist)
    step1_img = vis_base.copy()
    for c in candidate_columns:
        # Find continuous segments of black pixels in this column
        in_segment = False
        seg_start = None
        for y in range(height):
            if cleaned_img[y, c] == 0:  # Black pixel
                if not in_segment:
                    seg_start = y
                    in_segment = True
            else:  # White pixel
                if in_segment:
                    # Draw the segment we just finished
                    cv2.line(step1_img, (c, seg_start), (c, y-1), (255, 0, 0), 1)
                    in_segment = False
        # Draw final segment if we ended on black
        if in_segment:
            cv2.line(step1_img, (c, seg_start), (c, height-1), (255, 0, 0), 1)
    cv2.imwrite(f"{debug_folder}/step1_all_candidates.png", step1_img)
    print(f"  Step 1: Found {len(candidate_columns)} candidate columns")

    # Group consecutive columns into line segments
    line_segments = []
    line_segments_extents = []  # Store (x, top, bottom) for each segment
    if len(candidate_columns) > 0:
        current_segment = [candidate_columns[0]]

        for i in range(1, len(candidate_columns)):
            if candidate_columns[i] == candidate_columns[i-1] + 1:
                current_segment.append(candidate_columns[i])
            else:
                if len(current_segment) > 0:
                    line_segments.append(current_segment)
                current_segment = [candidate_columns[i]]

        if len(current_segment) > 0:
            line_segments.append(current_segment)

    # Calculate extents for each segment and prepare visualization
    step2_img = vis_base.copy()
    for segment in line_segments:
        line_x = int(np.mean(segment))
        # Find all continuous black segments for this line
        black_regions = set()  # Store all y-coordinates that have black pixels
        for x in segment:
            for y in range(height):
                if cleaned_img[y, x] == 0:
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

    # Analyze each line segment
    after_width_filter = []  # Store (x, top, bottom)
    after_height_filter = []  # Store (x, top, bottom, height)
    after_alignment_filter = []  # Store (x, top, bottom, is_single)
    after_curve_filter = []  # Store (x, top, bottom)
    valid_barlines = []

    for segment in line_segments:
        if len(segment) == 0:
            continue

        # Get the center x-coordinate of this line segment
        line_x = int(np.mean(segment))
        line_width = len(segment)

        # Find the vertical extent of this line
        line_pixels = []
        for x in segment:
            for y in range(height):
                if cleaned_img[y, x] == 0:
                    line_pixels.append(y)

        if len(line_pixels) == 0:
            continue

        line_top = min(line_pixels)
        line_bottom = max(line_pixels)
        line_height = line_bottom - line_top

        # Check if line width is consistent (within 1-2 pixels throughout)
        if line_width > 15:  # Too thick, probably not a barline
            continue

        after_width_filter.append((line_x, line_top, line_bottom))

        # Check if line height matches staff height criteria
        if line_height < min_line_height:
            continue

        after_height_filter.append((line_x, line_top, line_bottom, line_height))

        # Check if line aligns with staff boundaries
        valid_alignment = False
        is_single_staff = False

        for idx, (staff_top, staff_bottom) in enumerate(staff_positions):
            # Single staff barline
            tolerance = avg_staff_height * 0.15  # Increased tolerance to 15%
            if (abs(line_top - staff_top) < tolerance and
                abs(line_bottom - staff_bottom) < tolerance and
                min_line_height <= line_height <= max_single_staff_height):
                valid_alignment = True
                is_single_staff = True
                break

            # Double staff barline (extends to next staff)
            if idx < len(staff_positions) - 1:
                next_staff_bottom = staff_positions[idx + 1][1]
                if (abs(line_top - staff_top) < tolerance and
                    abs(line_bottom - next_staff_bottom) < tolerance and
                    line_height <= max_double_staff_height):
                    valid_alignment = True
                    is_single_staff = False
                    break

        if not valid_alignment:
            continue

        after_alignment_filter.append((line_x, line_top, line_bottom, is_single_staff))

        # For single staff height lines, check for curves at endpoints
        has_curve = False
        if is_single_staff:
            check_radius = 5  # pixels to check around endpoints

            # Check top endpoint
            for x_offset in range(-check_radius, check_radius + 1):
                check_x = line_x + x_offset
                if 0 <= check_x < width:
                    # Check pixels above the line top
                    for y_offset in range(1, check_radius):
                        check_y = line_top - y_offset
                        if 0 <= check_y < height:
                            # Look for horizontal black pixels (indicating a curve/note)
                            if cleaned_img[check_y, check_x] == 0:
                                # Check if this extends horizontally (curve indicator)
                                horizontal_count = 0
                                for hx in range(max(0, check_x - 3), min(width, check_x + 4)):
                                    if cleaned_img[check_y, hx] == 0:
                                        horizontal_count += 1
                                if horizontal_count >= 3:
                                    has_curve = True
                                    break
                    if has_curve:
                        break

            # Check bottom endpoint
            if not has_curve:
                for x_offset in range(-check_radius, check_radius + 1):
                    check_x = line_x + x_offset
                    if 0 <= check_x < width:
                        # Check pixels below the line bottom
                        for y_offset in range(1, check_radius):
                            check_y = line_bottom + y_offset
                            if 0 <= check_y < height:
                                # Look for horizontal black pixels
                                if cleaned_img[check_y, check_x] == 0:
                                    horizontal_count = 0
                                    for hx in range(max(0, check_x - 3), min(width, check_x + 4)):
                                        if cleaned_img[check_y, hx] == 0:
                                            horizontal_count += 1
                                    if horizontal_count >= 3:
                                        has_curve = True
                                        break
                        if has_curve:
                            break

        if has_curve:
            continue  # Skip this line as it curves into a note

        after_curve_filter.append((line_x, line_top, line_bottom))

        # Check that the line has consistent width throughout its height
        width_consistent = True
        sample_points = np.linspace(line_top, line_bottom, min(20, int(line_height))).astype(int)

        for sample_y in sample_points:
            # Count black pixels at this height within the segment
            black_count = 0
            for x in range(max(0, line_x - line_width), min(width, line_x + line_width + 1)):
                if cleaned_img[sample_y, x] == 0:
                    black_count += 1

            # Check if width varies too much
            if abs(black_count - line_width) > 3:  # Slightly more lenient
                width_consistent = False
                break

        if not width_consistent:
            continue

        # This is a valid barline!
        valid_barlines.append((line_x, line_top, line_bottom))

    # STEP 3: Show after width filter
    step3_img = vis_base.copy()
    for x, top, bottom in after_width_filter:
        draw_continuous_segments(step3_img, x, top, bottom, cleaned_img, (255, 255, 0), 2)
    cv2.imwrite(f"{debug_folder}/step3_after_width_filter.png", step3_img)
    print(f"  Step 3: {len(after_width_filter)} lines after width filter")

    # STEP 4: Show after height filter
    step4_img = vis_base.copy()
    for x, top, bottom, h in after_height_filter:
        draw_continuous_segments(step4_img, x, top, bottom, cleaned_img, (128, 0, 128), 2)
        cv2.putText(step4_img, f"{h:.0f}", (x+5, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f"{debug_folder}/step4_after_height_filter.png", step4_img)
    print(f"  Step 4: {len(after_height_filter)} lines after height filter (min: {min_line_height:.1f})")

    # STEP 5: Show after alignment filter
    step5_img = vis_base.copy()

    # First, draw the staff boundaries that we're aligning against
    for idx, (staff_top, staff_bottom) in enumerate(staff_positions):
        # Draw staff boundaries in bright green
        cv2.line(step5_img, (0, staff_top), (width, staff_top), (0, 255, 0), 2)
        cv2.line(step5_img, (0, staff_bottom), (width, staff_bottom), (0, 255, 0), 2)
        # Add staff number label
        cv2.putText(step5_img, f"Staff {idx+1}", (10, staff_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw tolerance zones (semi-transparent)
        tolerance = int(avg_staff_height * 0.15)
        overlay = step5_img.copy()
        cv2.rectangle(overlay, (0, staff_top - tolerance), (width, staff_top + tolerance),
                     (0, 255, 0), -1)
        cv2.rectangle(overlay, (0, staff_bottom - tolerance), (width, staff_bottom + tolerance),
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, step5_img, 0.9, 0, step5_img)

    # Then draw the lines that passed alignment filter
    for x, top, bottom, is_single in after_alignment_filter:
        color = (0, 128, 255) if is_single else (255, 128, 0)
        draw_continuous_segments(step5_img, x, top, bottom, cleaned_img, color, 2)

    cv2.imwrite(f"{debug_folder}/step5_after_alignment_filter.png", step5_img)
    print(f"  Step 5: {len(after_alignment_filter)} lines after alignment filter")

    # STEP 6: Show after curve filter
    step6_img = vis_base.copy()
    for x, top, bottom in after_curve_filter:
        draw_continuous_segments(step6_img, x, top, bottom, cleaned_img, (0, 255, 128), 2)
    cv2.imwrite(f"{debug_folder}/step6_after_curve_filter.png", step6_img)
    print(f"  Step 6: {len(after_curve_filter)} lines after curve filter")

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
    Create a visualization with measure areas highlighted.
    Returns a color image with measures outlined.
    """
    # Convert grayscale to color for visualization
    if len(cleaned_img.shape) == 2:
        vis_img = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = cleaned_img.copy()

    # Calculate staff positions
    no_of_staves = len(staff_lines) // 5
    staff_positions = []

    for i in range(no_of_staves):
        top_line = staff_lines[i * 5]
        bottom_line = staff_lines[i * 5 + 4]
        staff_positions.append((top_line, bottom_line))

    if len(measure_lines) < 2:
        return vis_img

    # Draw measure areas between consecutive barlines
    for i in range(len(measure_lines) - 1):
        x1 = measure_lines[i]
        x2 = measure_lines[i + 1]

        # Draw rectangles for each staff
        for staff_top, staff_bottom in staff_positions:
            # Draw semi-transparent rectangle for measure area
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (x1, staff_top), (x2, staff_bottom), (0, 255, 0), 2)

            # Add a light fill to the measure area
            cv2.rectangle(overlay, (x1, staff_top), (x2, staff_bottom), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.1, vis_img, 0.9, 0, vis_img)

            # Draw the border again to make it clearer
            cv2.rectangle(vis_img, (x1, staff_top), (x2, staff_bottom), (0, 255, 0), 2)

    # Draw the barlines themselves in red
    for x in measure_lines:
        cv2.line(vis_img, (x, 0), (x, cleaned_img.shape[0]), (0, 0, 255), 2)

    return vis_img

def preprocessing(inputfolder, outputfolder, fn, f):
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
    x1, y1, x2, y2 = boundary
    cur_symbol = cutted[y1-height_before:y2+1-height_before, x1:x2+1]

    # Clean and cut #
    cur_symbol = clean_and_cut(cur_symbol)
    cur_symbol = 255 - cur_symbol

    # Start prediction of the current symbol #
    feature = extract_hog_features(cur_symbol)
    label = str(model.predict([feature])[0])

    return get_target_boundaries(label, cur_symbol, y2)

def process_image(inputfolder, outputfolder, fn, f):
    cutted, ref_lines, lines_spacing = preprocessing(inputfolder, outputfolder, fn, f)

    last_acc = ''
    last_num = ''
    height_before = 0

    if len(cutted) > 1:
        f.write('{\n')


    for it in range(len(cutted)):
        f.write('[')
        is_started = False
        

        symbols_boundaries = segmentation(height_before, cutted[it])
        symbols_boundaries.sort(key = lambda x: (x[0], x[1]))
        
        for boundary in symbols_boundaries:
            label, cutted_boundaries = get_label_cutted_boundaries(boundary, height_before, cutted[it])

            if label == 'clef':
                is_started = True
            
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
                            last_acc=  ''
                            
                        if last_num != '':
                            text = f'\meter<"{text}/{last_num}">'
                            last_num =  ''
                        
                        not_dot = label != 'dot'
                        f.write(not_dot * ' ' + text)
            
        height_before += cutted[it].shape[0]
        f.write(' ]\n')
        
    if len(cutted) > 1:
        f.write('}')

def main():
    # Clean output folder before processing
    if os.path.exists(args.outputfolder):
        print(f"Cleaning output folder: {args.outputfolder}")
        shutil.rmtree(args.outputfolder)

    # Create fresh output folder
    os.makedirs(args.outputfolder, exist_ok=True)
    print(f"Created output folder: {args.outputfolder}\n")

    list_of_images = os.listdir(args.inputfolder)
    for _, fn in enumerate(list_of_images):
        # Open the output text file #
        file_prefix = fn.split('.')[0]
        f = open(f"{args.outputfolder}/{file_prefix}.txt", "w")

        # Process each image separately #
        try:
            process_image(args.inputfolder, args.outputfolder, fn, f)
        except Exception as e:
            print(e)
            print(f'{args.inputfolder}-{fn} has been failed !!')
            pass
        
        f.close()  
    print('Finished !!') 


if __name__ == "__main__":
    main()

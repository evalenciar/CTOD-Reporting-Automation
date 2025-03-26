# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:39:48 2025

Reads scale bar length and generates pixel to inches ratio for the image

@author: pcunha
"""

import cv2
import pytesseract
import numpy as np
from PIL import Image
import openpyxl


def read_tiff_pil(image_path):
    """
    Reads a TIFF image using PIL and converts it to OpenCV format.
    :param image_path: Path to the TIFF file.
    :return: OpenCV-compatible image (NumPy array) or None if reading fails.
    """
    try:
        # Open TIFF using PIL
        img = Image.open(image_path)

        # Convert to NumPy array (grayscale or color)
        img = np.array(img)

        # Convert RGB to BGR (if needed for OpenCV)
        if len(img.shape) == 3:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img
    except Exception as e:
        print(f"Error reading TIFF: {e}")
        return None

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pcunha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

#%% Add a section that saves the image as a JPG first! and crop

image_path = r"A:\Projects\Projects 101251-101275\101256 (P66 - BL01 Testing)\Metallurgical\CTOD\-6\101256-6B-1.tif"
img = read_tiff_pil(image_path)

ScaleBarImg = img[1500:1900, 1400:2550] # Crop the image to remove the bottom and right side buffers
cv2.imshow("Text Image",ScaleBarImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
text = pytesseract.image_to_string(ScaleBarImg)

#%% OCR getting the scale length

text = text.split("\n")[0]
text = text.split()
unit = text[1]
length = float(text[0])

#%% Image Processing for scale bar length

gray = cv2.cvtColor(ScaleBarImg, cv2.COLOR_BGR2GRAY) # convert to grayscale

_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY) # set any pixel below first value equal to second value (white)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the longest horizontal white bar
max_width = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > max_width:
        max_width = w

ratio = max_width/length #pixels per inch

print(f"Scale Bar Length: {length} {unit}")
print(f"Pixel Length of Scale Bar: {max_width} pixels")
print(f"Image Ratio: {ratio} pixels/in")


# # Optional: Display the result
# cv2.imshow("Threshold", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#%%
clone = img.copy()  # Copy to display user clicks
points = []  # Store user-selected points
VertMeasPoint = [] # store vertical measurement points for post processing image
RealDistances = [] # store vertical measurements
mode = "horizontal"  # Track whether selecting horizontal or vertical points
divided_x_positions = []  # Store positions for 9-sectioned lines

def draw_text_with_background(img, text, position, font_scale=1, thickness=2):
    """Draw text with a white background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x, text_y = position
    cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
    cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness)

def select_points(event, x, y, flags, param):
    """ Callback function for selecting points and drawing measurements. """
    global mode, divided_x_positions, points, VertMeasPoint, clone, RealDistances

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  # Store the clicked point
        VertMeasPoint.append((x,y))
        # Draw a red dot at the selected point
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)

        # Step 1: Horizontal mode - Get 2 points and divide into 9 sections
        if mode == "horizontal" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Ensure left-to-right order
            if x1 > x2:
                x1, x2 = x2, x1
            B = x2 - x1
            segment_length = B / 10

            # Draw 9 vertical white lines on the mask (for removal later)
            divided_x_positions = [int(x1 + i * segment_length) for i in range(1, 10)]  # 9 divisions
            for x_pos in divided_x_positions: 
                # Draw on clone for the visual guide
                cv2.line(clone, (x_pos, 0), (x_pos, clone.shape[0]), (255, 255, 255), 2)  

            print(f"Horizontal segment length: {B} pixels, Divided into 9 sections.")
            
            # Clear points for vertical measurements
            points = []
            VertMeasPoint = []
            mode = "vertical"  # Switch to vertical selection mode
            
        # Step 2: Vertical mode - Require two NEW clicks for each measurement
        elif mode == "vertical" and len(points) == 3:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            
            # Compute vertical distance
            vertical_distance1 = abs(y2 - y1)
            real_distance1 = vertical_distance1 / ratio  # Convert using scale factor
            RealDistances.append(real_distance1)
            
            vertical_distance2 = abs(y3 - y1)
            real_distance2 = vertical_distance2 / ratio  # Convert using scale factor
            RealDistances.append(real_distance2)
            
            # Draw vertical line (keeps the line intact even after clearing white lines)
            mid_x = (x1 + x2 + x3) // 3
            cv2.line(clone, (mid_x, y1), (mid_x, y2), (255, 0, 0), 2)

            # Draw vertical line (keeps the line intact even after clearing white lines)
            cv2.line(clone, (mid_x, y2), (mid_x, y3), (255, 0, 0), 2)

            # Display distance with background
            mid_y1 = (y1 + y2) // 2
            draw_text_with_background(clone, f"{real_distance1:.4f}", (mid_x, mid_y1))
            
            mid_y2 = (y3 + y2) // 2
            draw_text_with_background(clone, f"{real_distance2:.4f}", (mid_x, mid_y2))

            print(f"Vertical distance: {vertical_distance1} pixels ({real_distance1:.4f} real units)")

            # Reset points to allow more vertical measurements
            points = []

        # Refresh the display with updated image
        cv2.imshow("Measure Distances", clone)

# Display the initial image
cv2.namedWindow("Measure Distances", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Measure Distances", clone)
cv2.setMouseCallback("Measure Distances", select_points)

# Keep the window open until the user presses 'Enter' to erase lines or 'q' to quit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to exit
        break
    elif key == 13:  # Enter key pressed (key code 13)
        j = 0
        Old_y1 = 0
        Old_y2 = 0
        print("Preston's Final Masterpiece:")
        draw_text_with_background(img, "Pre Crack",(50, 300), 2, 3)
        draw_text_with_background(img,"J Whatevs",(2250, 300), 2, 3)
        for i in range(len(VertMeasPoint)-1):
            if i%3 == 0:
                
                mid_x =( VertMeasPoint[i][0] + VertMeasPoint[i+1][0] + VertMeasPoint[i+1][0]) //3
                
                mid_y1 = (VertMeasPoint[i][1] + VertMeasPoint[i+1][1]) //2
                mid_y2 = (VertMeasPoint[i+1][1] + VertMeasPoint[i+2][1]) //2
                
                if Old_y1-mid_y1 < 50:
                    mid_y1 = mid_y1 +70
                    
                if Old_y2-mid_y2 < 5:
                    mid_y1 = mid_y1 + 10
                
                Old_y1 = mid_y1
                Old_y2 = mid_y2
                
                cv2.line(img,(mid_x,VertMeasPoint[i][1]),(mid_x,VertMeasPoint[i+1][1]), (0,0,255),2)
                
                cv2.line(img,(mid_x,VertMeasPoint[i+1][1]),(mid_x,VertMeasPoint[i+2][1]), (0,0,255),2)
                
                cv2.line(img,(mid_x-20,VertMeasPoint[i][1]),(mid_x+20,VertMeasPoint[i][1]), (0,0,255),2)
                cv2.line(img,(mid_x-20,VertMeasPoint[i+1][1]),(mid_x+20,VertMeasPoint[i+1][1]), (0,0,255),2)
                cv2.line(img,(mid_x-20,VertMeasPoint[i+2][1]),(mid_x+20,VertMeasPoint[i+2][1]), (0,0,255),2)
                
        for i in range(len(RealDistances)):
            if i%2 == 0:
                draw_text_with_background(img,f"{RealDistances[i]:.4f} in",(50, 400+i*25))
            else:
                draw_text_with_background(img,f"{RealDistances[i]:.4f} in",(2250, 400+i*25))
            
        
        cv2.imshow("Measure Distances", img)

cv2.destroyAllWindows()

#%%
print("heyo")
# Load the existing workbook
file_path = r"A:\Projects\09_R&D\100001 (Materials - CTOD Reporting Automation)\Client Docs\Crack Check.xlsx"  # Make sure this file exists
workbook = openpyxl.load_workbook(file_path)

# Select the active worksheet (or specify a sheet by name)
sheet = workbook.active  # Or use workbook["SheetName"] if you know the sheet name

# Example: Assign values to specific cells
sheet["B2"] = RealDistances[0]
sheet["B3"] = RealDistances[2]
sheet["B4"] = RealDistances[4]
sheet["B5"] = RealDistances[6]
sheet["B6"] = RealDistances[8]
sheet["B7"] = RealDistances[10]
sheet["B8"] = RealDistances[12]
sheet["B9"] = RealDistances[14]
sheet["B10"] = RealDistances[16]

sheet["E2"] = RealDistances[1]
sheet["E3"] = RealDistances[3]
sheet["E4"] = RealDistances[5]
sheet["E5"] = RealDistances[7]
sheet["E6"] = RealDistances[9]
sheet["E7"] = RealDistances[11]
sheet["E8"] = RealDistances[13]
sheet["E9"] = RealDistances[15]
sheet["E10"] = RealDistances[17]

workbook.save(file_path)

print("Values written successfully!")












































































# Space 
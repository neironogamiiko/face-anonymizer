##########################################
#    Simple face anonymizer for photo    #
##########################################

import cv2; import os
import mediapipe as mp

##########################################
#           Read image by path           #
##########################################

image_path = "D:\\images\\test_image.jpg"
original_image = cv2.imread(image_path)
image_height, image_width, _ = original_image.shape

##########################################
#             Face detection             #
##########################################

face_detection = mp.solutions.face_detection

with face_detection.FaceDetection(model_selection=0, min_detection_confidence=.5) as face_detection:

    # model_selection:
    #   0 (Short-range model):
    #   Optimized for detecting faces within a closer range, typically up to 2 meters from the camera.
    #
    #   1 (Full-range model):
    #   Optimized for detecting faces at a greater distance, typically up to 5 meters from the camera.

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    process_result = face_detection.process(rgb_image)

    # Look at detection result (location_data, relative_bounding_box, relative_keypoint, etc):
    # print(process_result.detections)

    try:
        for detection in process_result.detections:
            location_data = detection.location_data
            bounding_box = location_data.relative_bounding_box
            x, y, width, height = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height

            x = int(x * image_width); y = int(y * image_height)
            width = int(width * image_width); height = int(height * image_height)

            # Drawing rectangle around face:
            # face_bounding_box = cv2.rectangle(original_image, (x, y), (x + width, y + height), (0,0,255), 5)

            # Bluring by replacement face with blured rectangle
            copy_original_image = original_image.copy()
            copy_original_image[y:y+height, x:x+width, :] = cv2.blur(original_image[y:y+height, x:x+width, :], (50,50))
    except TypeError:
        print("None Type Error: There is no face on image.")

cv2.imshow('Image', copy_original_image)
cv2.waitKey(0); cv2.destroyAllWindows()

##########################################
#           Read image by path           #
##########################################

output_path = "D:\\images\\Output"

if not os.path.exists(output_path):
    os.makedirs(output_path)

cv2.imwrite(os.path.join(output_path, 'BluredImage.jpg'), copy_original_image)
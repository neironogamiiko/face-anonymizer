import cv2; import os
import mediapipe as mp
import argparse

def anonymize_all_faces(original_image, face_detection):
    image_height, image_width, _ = original_image.shape

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

            x = int(x * image_width);
            y = int(y * image_height)
            width = int(width * image_width);
            height = int(height * image_height)

            # Drawing rectangle around face:
            # face_bounding_box = cv2.rectangle(original_image, (x, y), (x + width, y + height), (0,0,255), 5)

            # Bluring by replacement face with blured rectangle
            copy_original_image = original_image.copy()
            copy_original_image[y:y + height, x:x + width, :] = cv2.blur(original_image[y:y + height, x:x + width, :],
                                                                         (50, 50))
    except Exception as error:
        print(f"Error: {error}")
        return None

    return copy_original_image


##########################################
#           Read image by path           #
##########################################

args_for_anonymizer = argparse.ArgumentParser()
args_for_anonymizer.add_argument("--mode", default="video")
args_for_anonymizer.add_argument("--filePath", default="D:/images/testVideo.mp4")
# args_for_anonymizer.add_argument("--mode", default="image")
# args_for_anonymizer.add_argument("--filePath", default="D:/images/text.png")
args_for_anonymizer = args_for_anonymizer.parse_args()

output_path = "D:/images/output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

##########################################
#             Face detection             #
##########################################

face_detection = mp.solutions.face_detection
with face_detection.FaceDetection(model_selection=0, min_detection_confidence=.7) as face_detection:
    if args_for_anonymizer.mode in ["image"]:
        original_image = cv2.imread(args_for_anonymizer.filePath)
        anonymizer_result = anonymize_all_faces(original_image, face_detection)

        if anonymizer_result is not None:
            cv2.imshow('Image', anonymizer_result)
            cv2.waitKey(0); cv2.destroyAllWindows()

            ##########################################
            #          Write image by path           #
            ##########################################

            cv2.imwrite(os.path.join(output_path, 'BluredImage.jpg'), anonymizer_result)

    elif args_for_anonymizer.mode in ["video"]:
        video = cv2.VideoCapture(args_for_anonymizer.filePath)
        ret, frame = video.read()

        output_path = "D:\\images\\Output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file_path = os.path.join(output_path, "anonymeVideo.mp4")
        output_video = cv2.VideoWriter(output_file_path,
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = anonymize_all_faces(frame, face_detection)
            output_video.write(frame)
            ret, frame = video.read()

        video.release(); output_video.release()
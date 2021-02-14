from imageai.Detection import ObjectDetection
import os

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
detector.loadModel()


# process image
directory = 'images'
file_names = os.listdir(directory)
file_names.sort()
for file in file_names:
    print(file)
    detections = detector.detectObjectsFromImage(input_image="images/" + file,
                                                 output_image_path="images/done/" + file,
                                                 minimum_percentage_probability=30)


# process video
# detections = detector.detectObjectsFromVideo(input_image="image/video.mp4",
# output_image_path="videonew", frames_per_second=5, log_progress=True)

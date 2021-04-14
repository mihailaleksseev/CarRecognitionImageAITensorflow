from imageai.Detection import ObjectDetection
import time
import os

print('[START MODEL LOAD] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

detector = ObjectDetection()

""" resnet50_coco_best_v2.1.0.h5
    - модель весит 152 mb
    - время загрузки модели 3-4 секунды
    - время процессинга первых изображений 20-30 секунд, после 4го скорость стала составлять около секудны
    - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
"""
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
# folder = 'done_resnet50/'


""" yolo-tiny.h5
    - модель весит 248 mb
    - время загрузки модели 2 секунды
    - время процессинга изображения стабильно 3-4 секунды
    - приемлимо находит авто, было обнаружено на всех фото, но контур не всегда корректны
    - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
"""
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath("yolo.h5")
# folder = 'done_yolo/'


""" yolo.h5
    - время загрузки модели меньше секунды 
    - время процессинга изображения стабильно 3-4 секунды
    - очень плохо находит авто, на 2х из 7 фото авто обнаружено не было
    - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
"""
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5")
folder = 'done_yolotiny/'

detector.loadModel()

print('[END MODEL LOAD] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

# process image
directory = 'images'
file_names = os.listdir(directory)
file_names.sort()
for file in file_names:
    print('\n [START FILE' + file + ' READ ] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))
    detections = detector.detectObjectsFromImage(input_image="images/" + file,
                                                 output_image_path="images/" + folder + file,
                                                 minimum_percentage_probability=30)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

    print('[END FILE' + file + ' READ ] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

# process video
# detections = detector.detectObjectsFromVideo(input_image="image/video.mp4",
# output_image_path="videonew", frames_per_second=5, log_progress=True)

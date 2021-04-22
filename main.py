from imageai.Detection import ObjectDetection
import time
import os

print('[START MODEL LOAD] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

# """ yolo-tiny.h5
#     - модель весит 248 mb
#     - время загрузки модели 2 секунды
#     - время процессинга изображения стабильно 3-4 секунды
#     - приемлимо находит авто, было обнаружено на всех фото, но контур не всегда корректны
#     - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
# """
# detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath("yolo.h5")
# folder = 'done_yolo/'


# """ yolo.h5
#     - время загрузки модели меньше секунды
#     - время процессинга изображения стабильно 3-4 секунды
#     - очень плохо находит авто, на 2х из 7 фото авто обнаружено не было
#     - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
# """
# detector = ObjectDetection()
# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath("yolo-tiny.h5")
# folder = 'done_yolotiny/'

""" resnet50_coco_best_v2.1.0.h5
    - модель весит 152 mb
    - время загрузки модели 3-4 секунды
    - время процессинга первых изображений 20-30 секунд, после 4го скорость стала составлять около секудны
    - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
"""
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
folder = 'done_resnet50/'

# """ resnet50_coco_best_v2.1.0.h5 с параметрами car/truck/bus
#     - модель весит 152 mb
#     - время загрузки модели 3-4 секунды
#     - визуально время распознавания занимает около секунды
#     - запуск осуществлялся на macbook pro 2017 i7, 16gb, Radeon Pro 555X 4gb
# """
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
# folder = 'done_resnet50_custom_objects/'
# custom = detector.CustomObjects(car=True, truck=True, bus=True)

detector.loadModel()
print('[END MODEL LOAD] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))
# process image
file_names = os.listdir('images/to_recognize')
file_names.sort()
for file_name in file_names:
    # отсеивание системных файлов
    if file_name[:4] == 'car_':

        print('\n' + '[START FILE ' + file_name + ' READ ] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

        # """
        #     Для моделей с дефолтными параметрами
        # """
        # detections = detector.detectObjectsFromImage(custom_objects=custom,
        #                                                    input_image="images/to_recognize/" + file_name,
        #                                                    output_image_path="images/" + folder + file_name,
        #                                                    minimum_percentage_probability=30)

        """
            Для моделей с дефолтными параметрами
        """
        detections = detector.detectObjectsFromImage(input_image="images/to_recognize/" + file_name,
                                                     output_image_path="images/" + folder + file_name,
                                                     minimum_percentage_probability=30)

        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
            print("--------------------------------")

        print('[END FILE ' + file_name + ' READ ] ' + time.strftime("%d:%m:%Y-%H:%M:%S"))

# process video
# detections = detector.detectObjectsFromVideo(input_image="image/video.mp4",
# output_image_path="videonew", frames_per_second=5, log_progress=True)

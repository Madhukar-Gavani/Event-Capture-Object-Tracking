Python: 3.8.10
openCV: 4.5.2
scipy: 1.7.1
numpy: 1.19.5

tensorflow model: ssd_mobilenet_v3_large_coco_2020_01_14
object tracking algorithm: using eculidian distance method

source: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
Object tracking: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

completed:
1: person detection with threshold confidance = 0.61(in video)
2: person tracking works well upto max of three persons
4: assign entry time to each person when the video starts

problems:
1: person detection fails sometime(random)
2: tracking not efficient(has even too many problems)

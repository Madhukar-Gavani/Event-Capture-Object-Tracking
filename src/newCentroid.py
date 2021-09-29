from collections import OrderedDict
from datetime import datetime
import cv2 as cv
import numpy as np
from scipy.spatial import distance

class CentroidAlgorithm:
    def __init__(self, maxNumberOfDisapper=30):
        self.nextObject = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxNumberOfDisapper = maxNumberOfDisapper
    
    def __del__(self):
        del self.objects
        del self.disappeared
        print("all objects deregisterd")
    
    def register(self, centroidValues):
        self.registerEvent(centroidValues, None)    

    def registerEvent(self, centroidValues, EntryTime):
        print("object registered at ",self.nextObject, ": ", centroidValues)
        self.objects[self.nextObject] = {"centroid":centroidValues, "entrytime": EntryTime, "isUpdated": False}

        self.disappeared[self.nextObject] = 0
        self.nextObject += 1
    

    def deregister(self, objectID):
        print("object deregistered at ", objectID,": ", self.objects[objectID])
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rectange, EntryTime):
        # check if length of box is 0 if this then count number of occurence
        # this disappearence and count disappearence to deregister the object
        if len(rectange) == 0:
            # get every object 
            for objectId in list(self.objects.keys()):
                self.disappeared[objectId] += 1
                # check agains the maxdisappearence of the objects
                # if so deregister that object
                if self.disappeared[objectId] > self.maxNumberOfDisapper:
                    self.deregister(objectId)
            
            # return the objects
            return self.objects


        # if not that case get the rectange calculate the distance from previes 
        # first store the number of centroids in CurrentCentroids
        # make its shape of rectange
        currentCentroids = np.zeros((len(rectange), 2), dtype="int")
        # get coordinates of the box
        for (i, (startX, startY, height, width)) in enumerate(rectange):
            # calculate centroid of the box frame
            X_centroid = int((2 * startX + height) / 2.0)
            Y_centroid = int((2 * startY + width) / 2.0)
            # push into currentCentroids for further use
            currentCentroids[i] = (X_centroid, Y_centroid)

        # initial condition check againts the updated centroid array 
        # register the objects upto how many box we have
        if len(self.objects) == 0:
            for i in range(0, len(currentCentroids)):
                self.register(currentCentroids[i])
                if self.objects[self.nextObject-1]["isUpdated"] == False:
                    self.objects[self.nextObject-1]["entrytime"] = EntryTime
                    self.objects[self.nextObject-1]["isUpdated"] = True
        
        # if not get the eculidean distance between the previous centroid of frame from objects[objectID] to currentCentroids
        else:
            objectIDs = list(self.objects.keys())
            objectsValue = list(self.objects.values())
            # print(objectsValue)
            objectCentroids = [list(value["centroid"]) for value in objectsValue]
            # print(objectCentroids)
            # find eculidean distance between previous frame centroid to current frame centroid
            eculideanDistance = distance.cdist(np.array(objectCentroids), currentCentroids)
            # get the minimum distance between two centeroids
            # now eculideanDistance is of the size len(currentCentroids) X len(currentCentroids)
            # so in this every ith row is first input's ith array
            # and column represents second input's ith array
            # axis=1 for rowise check

            # first find overall minimum
            rows = eculideanDistance.min(axis=1).argsort()
            # since we get row we want to find column to get perticular index
            cols = eculideanDistance.argmin(axis=1)[rows]

            # keep tack of which of the column we examined
            usedRows = set() #as set doesn't duplicate
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # if alredy examined do nothing
                if row in usedRows or col in usedCols:
                    continue
                
                # else update the centroid
                objectID = objectIDs[row] # since objectCentroids is first row argument
                self.objects[objectID]["centroid"] = currentCentroids[col] # since currentCentroids is column row argument
                self.disappeared[objectID] = 0
                
                # update this row and column to indicate we examined
                usedRows.add(row)
                usedCols.add(col)
            
            # there are may be some unused rows and columns
            unusedRows = set(range(0, eculideanDistance.shape[0])).difference(usedRows)
            unusedCols = set(range(0, eculideanDistance.shape[1])).difference(usedCols)

            # check the number of object centroid and current centroid
            # if it is greater than or equal to current centroid 
            # we need to check and see some of the object disapperead
            if len(np.array(objectCentroids)) >= len(currentCentroids):
                # check in unused rows 
                for row in unusedRows:
                    # get the objectId and increment the disappreance
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check for maximum to deregsiter the object
                    if self.disappeared[objectID] > self.maxNumberOfDisapper:
                        self.deregister(objectID)

            # if object centroid is less than currentcentroid then new object has 
            # arrieved register the object
            else:
                for col in unusedCols:
                    self.register(currentCentroids[col])
                    if self.objects[self.nextObject - 1]["isUpdated"] == False:
                        self.objects[self.nextObject-1]["entrytime"] = EntryTime 
                        self.objects[self.nextObject-1]["isUpdated"] = True
        
        return self.objects  
                
class EventCapture:
    def __init__(self):
        self.starttimer = None
    
    def startTimer(self):
        self.starttimer = datetime.now()
        
    
    def event(self):
        return (datetime.now() - self.starttimer)


# Set the model
model = "../ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
config = "../config_file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
DL_model = cv.dnn_DetectionModel(model, config)

# Set the labels
labels = []
with open("../labels.txt") as labelFile:
    labels = labelFile.read().split("\n")
labels.pop()

DL_model.setInputSize(320, 320)
DL_model.setInputScale(1.0/127.0)
DL_model.setInputMean((127.5, 127.5, 127.5))
DL_model.setInputSwapRB(True)

def resizeScaleFrame(frame, scale=0.25):
    height = frame.shape[0]
    width = frame.shape[1]
    
    scale_dimenstion = (int(width * scale), int(height * scale))
    model_dimenstion = (320, 320)
    scale_img = cv.resize(frame, scale_dimenstion, interpolation=cv.INTER_CUBIC)
    model_img = cv.resize(frame, model_dimenstion, interpolation=cv.INTER_CUBIC)
    return [scale_img, model_img]



# capture the video
capture_video = cv.VideoCapture(0)
timer = EventCapture()
timer.startTimer()
centroidAlgo = CentroidAlgorithm()

def detectTheObject(frame):
    class_indexes, confidence_levels, border_boxes = DL_model.detect(frame)
    rectange = []
    if len(class_indexes) > 0:
        for class_index, confidence, border_box in zip(class_indexes.flatten(), confidence_levels.flatten(), border_boxes):
            
            #check only persons and make border for them
            if confidence > 0.61:
                if class_index != 1:
                    continue
                print("{}, {}, {}".format(class_indexes[:,0], border_boxes[:], confidence_levels[:]))
                rectange.append(border_box.astype("int"))
                cv.rectangle(frame, border_box, (255,0, 0), 2)
                cv.putText(frame, labels[class_index - 1], (border_box[0], border_box[1]), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color=(0,255,255),thickness=1)

        
        objects = centroidAlgo.update(rectange, EntryTime=timer.event())
        for (ObjectID, info) in objects.items():
            # print(info["entrytime"])
            text = "ID {}, st {}, T {}".format(ObjectID, timer.starttimer.strftime("%S.%f"), str(info["entrytime"]).split(":")[2])
            cv.putText(frame, text, (info["centroid"][0], info["centroid"][1]), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color=(0,255,255),thickness=1)
            cv.circle(frame,(info["centroid"][0], info["centroid"][1]), 4, (255,0,0), -1)
    return frame

while True:
    isTrue, frame = capture_video.read()

    if not isTrue:
        print("Error!! unable to read the frame")
        break
    
    #resize the frame and show as video
    # new_frame = resizeScaleFrame(frame)
    # scaled_frame = detectTheObject(new_frame[0])
    # cv.imshow("test", new_frame[0])
    mapped_frame = detectTheObject(frame)
    # cv.imshow("scaled_video", scaled_frame)
    cv.imshow("mapped_video", mapped_frame)

    if cv.waitKey(20) & 0xFF == ord("s"):
        break

capture_video.release()
cv.destroyAllWindows()

cv.waitKey(0)
from time import sleep
import cv2
import argparse
import sys
import numpy as np
import os.path
from glob import glob
#from PIL import image
frame_count = 0             # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out=0           # used in post process loop, to get the no of specified class value.
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
winName = 'Yolo object detection in OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
for fn in glob('images/*.jpg'):
    img = cv2.imread(fn)
    #frame_count =0
    #img = cv2.imread("room_ser.jpg")
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

# Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

# Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
            # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

            # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   # print(indexes)
    count=0
    motor=0
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #print(label)
            if label=="person":
                color = colors[i]
                #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                count=count+1
            if label=="motorbike":
                motor=1
    if motor==1:
        print("\t Person= ")
        print(count)
    else:
        print("\t No motorbike\t Person= ")
        print(count)   
   # cv2.imshow("Image", img)

    #cv2.waitKey(400)

#detection

# Load names of classes1
    classesFile = "obj.names";
    classes1 = None
    with open(classesFile, 'rt') as f:
        classes1 = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3-obj.cfg";
    modelWeights = "yolov3-obj_2400.weights";

    net1 = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the names of the output layers
    def getOutputsNames(net1):
        # Get the names of all the layers in the network
        layersNames = net1.getLayerNames()
        # Get the names of the output layers, j.e. the layers with unconnected outputs
        return [layersNames[j[0] - 1] for j in net1.getUnconnectedOutLayers()]


    # Draw the predicted bounding box
    def drawPred(classId, conf, left, top, right, bottom):

        global frame_count
    # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label1 = '%.2f' % conf
        # Get the label1 for the class name and its confidence1
        if classes1:
            assert(classId < len(classes1))
            label1 = '%s:%s' % (classes1[classId], label1)

        #Display the label1 at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        #print(label1)            #testing
        #print(labelSize)        #testing
        #print(baseLine)         #testing

        label_name,label_conf = label1.split(':')    #spliting into class & confidance. will compare it with person.
        if label_name == 'Helmet':
                                            #will try to print of label1 have people.. or can put a counter to find the no of people occurance.
                                        #will try if it satisfy the condition otherwise, we won't print the boxes1 or leave it.
            cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label1, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1


        #print(frame_count)
        if(frame_count> 0):
            return frame_count




    # Remove the bounding boxes1 with low confidence1 using non-maxima suppression
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out=0
        classIds = []
        confidences1 = []
        boxes1 = []
        # Scan through all the bounding boxes1 output from the network and keep only the
        # ones with high confidence1 scores1. Assign the box's class label1 as the class with the highest score.
        classIds = []               #have to fins which class have hieghest confidence1........=====>>><<<<=======
        confidences1 = []
        boxes1 = []
        for out in outs:
            for detection in out:
                scores1 = detection[5:]
                classId = np.argmax(scores1)
                confidence1 = scores1[classId]
                if confidence1 > confThreshold:
                    center_x1 = int(detection[0] * frameWidth)
                    center_y1 = int(detection[1] * frameHeight)
                    width1 = int(detection[2] * frameWidth)
                    height1 = int(detection[3] * frameHeight)
                    left = int(center_x1 - width1 / 2)
                    top = int(center_y1 - height1 / 2)
                    classIds.append(classId)
                    #print(classIds)
                    confidences1.append(float(confidence1))
                    boxes1.append([left, top, width1, height1])

        # Perform non maximum suppression to eliminate redundant overlapping boxes1 with
        # lower confidences1.
        indices = cv2.dnn.NMSBoxes(boxes1, confidences1, confThreshold, nmsThreshold)
        count_person=0 # for counting the classes1 in this loop.
        for j in indices:
            j = j[0]
            box = boxes1[j]
            left = box[0]
            top = box[1]
            width1 = box[2]
            height1 = box[3]
                   #this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can calculate it.
            frame_count_out = drawPred(classIds[j], confidences1[j], left, top, left + width1, top + height1)
             #increase test counter till the loop end then print...

            #checking class, if it is a person or not

            my_class='Helmet'                   #======================================== mycode .....
            unknown_class = classes1[classId]

            if my_class == unknown_class:
                count_person += 1
        #if(frame_count_out > 0):
        print("Helmet count= ")
        print(frame_count_out)


        if count_person >= 1:
            path = 'test_out/'
            frame_name=os.path.basename(fn)             # trimm the path and give file name.
            cv2.imwrite(str(path)+frame_name, frame)     # writing to folder.
            #print(type(frame))
            cv2.imshow('img',frame)
            cv2.waitKey(800)


        #cv2.imwrite(frame_name, frame)
                                               #======================================mycode.........

    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    #fn="img0239.jpg"
    frame = cv2.imread(fn)
    #for fn in glob('images/*.jpg'):
    #    frame = cv2.imread(fn)
    frame_count =0

        # Create a 4D blob1 from a frame.
    blob1 = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
    net1.setInput(blob1)

        # Runs the forward pass to get output of the output layers
    outs = net1.forward(getOutputsNames(net1))

        # Remove the bounding boxes1 with low confidence1
    postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net1.getPerfProfile()
    #print(t)
    label1 = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    #print(label1)
    cv2.putText(frame, label1, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #print(label1)

    #detection
cv2.destroyAllWindows()

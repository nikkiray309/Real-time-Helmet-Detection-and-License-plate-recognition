# Identifying License plates of Non-Helmet Riders
# Importing Required Libraries
import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model

# Allow TensorFlow to use GPU
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Read the network weights from yolo files. These 'yolo.weights' is the file that we trained just to detect bikes and number plates
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg") # Bikes, License Plate Detection

# Setting the GPU and OpenCV with CUDA Backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Loading the Model
model = load_model('helmet-nonhelmet_cnn.h5')
print('--------------------')
print('CNN Model Loaded !!!')
print('--------------------')

# Processing
cap = cv2.VideoCapture('video.mp4')
COLORS = [(0,255,0),(0,0,255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output.mp4', fourcc, 5,(888,500))


def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
		pass


ret = True
k = 0
try:
    while ret:

        ret, img = cap.read() # returns a tuple where the first element is a boolean that indicates whether a frame has been grabbed or not
        img = imutils.resize(img, height=500)
    
        height, width = img.shape[:2]

        # create a blob from the image and to use this blob as input to our network
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
    
        confidences = []
        boxes = []
        classIds = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                color = [int(c) for c in COLORS[classIds[i]]]
            
                if classIds[i]==0: # Bike
                    helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4, max(0,x):max(0,x)+max(0,w)]
                else: # Number Plate
                    x_h = x-60
                    y_h = y-350
                    w_h = w+100
                    h_h = h+100

                    l_r = img[y:y+h, x:x+w] # License plate image

                    if y_h>0 and x_h>0:
                        h_r = img[y_h:y_h+h_h , x_h:x_h +w_h] # Helmet image
                        c = helmet_or_nohelmet(h_r)

                        if c == 1:
                            if not os.path.exists("c:\Python_Projects\Hel-Lic\LicensePlates"):
                                os.makedirs("LicensePlates")
                        
                            os.chdir("c:\Python_Projects\Hel-Lic\LicensePlates")
                        
                            try:
                                cv2.imwrite(str(k)+'.png', l_r) # Save the License plate img
                                k += 1
                            except:
                                pass
                            os.chdir("..")
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 7) # Bounding box(Red) for License plate

                        cv2.putText(img,['Helmet','No-Helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 2)             
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10) # Bounding Box(Blue) of Helmet

        # Write the output images as a video
        writer.write(img)
        cv2.imshow("Image", img)

        # Break the code if someone hits the ESC key
        if cv2.waitKey(1) == 27:
            break

    writer.release()
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except:
    print("Detection and Idetification is done !! Check LicensePlates folder")
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math   
from cvzone.ClassificationModule import Classifier
from collections import Counter
import pyperclip                                                

root = tk.Tk()
root.title("Sign Language")
canvas = tk.Canvas(root, width=640, height=700)
canvas.pack()

cap = cv2.VideoCapture("http://192.168.40.160:8080/video")
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
counter = 0
key = cv2.waitKey(1)

#labels = ["ROCK","Done"]

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


def on_key_press(event):
    if event.char == 's':
        # print("i am working") 
        pass    


list1 = []
add_list = []

def btn_event(who=0):
    
    if who != 0:
        if len(list1) <= 5: 
            list1.append(who)
        else:
            global lable_list
            lable_list = list1.copy()
            
            counter = Counter(lable_list)
            most_common_item, count = counter.most_common(1)[0]
            most = lable_list.count(most_common_item)

      
            if most ==6 or most == 4 :
                label3.place_forget()
                label5.place(x=10, y=10)
            else:
                label3.place(x=10, y=10)
                label5.place_forget()

            list1.clear()

        
            

def me(x=0):

    if x == 0:
        counter = Counter(lable_list)
        most_common_item, count = counter.most_common(1)[0]

        add_list.append(most_common_item)
        result = ''.join(add_list)
        
    if x == 1:
        add_list.clear()
        result = ""

    label1.config(text=f"{result}")

def better():
    print("Better !!!"+lable_list)

def copy_text():
        text_to_copy = label1.cget("text")
        pyperclip.copy(text_to_copy)
        label2.place(x=220, y=500)
       
def update():
    
    success, img = cap.read()
    imgCopy = img.copy()
    hands, img = detector.findHands(img)
    ch = "s"

    if success:
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.photo = photo  

        if hands :
            label.place_forget()
            label3.place_forget()
            label5.place_forget()

            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            imgCropShape = imgCrop.shape
    
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                # imgResizeShape = imgResize. shape
                widthGape = math.ceil((imgSize - wCal)/2)
                imgWhite[:,widthGape:wCal+widthGape] = imgResize
                pridiction, index = classifier.getPrediction(imgWhite,draw=False)
                # print(pridiction,index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                # imgResizeShape = imgResize. shape
                widthGape = math.ceil((imgSize - hCal)/2)
                imgWhite[widthGape:hCal+widthGape,:] = imgResize
                pridiction, index = classifier.getPrediction(imgWhite,draw=False)
                
            
            cv2.rectangle(imgCopy,(x-offset,y-offset-50),(x+offset+w,y-offset),(255,0,255),cv2.FILLED)
            cv2.putText(imgCopy,labels[index],(x,y-30),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,255,255),4)
            cv2.rectangle(imgCopy,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)



            btn_event(labels[index])
            rgb_frame = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.photo = photo 
            root.bind('<KeyPress>', on_key_press)
            lambda:me(0)
        else:
            
            label.place(x=220, y=200)
            label3.place(x=10, y=10)
            
            

    root.after(5, update) 
      
label = tk.Label(root, text="Hand Not Found", font=("Helvetica", 20), fg="blue", height=2)
label2 = tk.Label(root, text="Text Copied!", font=("Helvetica", 17), fg="red")

label3 = tk.Label(root, text="For Better Result Bring Your Hand Close To The Camera !", font=("Helvetica", 17), bg="black",fg="white")
label5 = tk.Label(root, text="Doing Great !!!", font=("Helvetica", 17), bg="black",fg="white")

label1 = tk.Label(root, text="", font=("Helvetica", 20), fg="white", height=1, bg='blue')
label1.place(x=12,y=545)

btn_add = tk.Button(root, text="Add",font=("Helvetica", 20),command=lambda:me(0),bd=4,width=8,justify="center")
btn_add.place(x=323,y=600)

btn_clear = tk.Button(root, text="Clear",font=("Helvetica", 20),command=lambda:me(1),bd=4,width=8,justify="center")
btn_clear.place(x=480,y=600)

btn_copy = tk.Button(root, text="Copy",font=("Helvetica", 20),command=lambda:copy_text(),bd=4,width=8,justify="center")
btn_copy.place(x=100,y=600)

update()


# Open the webcam
cap = cv2.VideoCapture(0)

# Set the video codec and properties
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



root.mainloop()

cap.release()
cv2.destroyAllWindows()



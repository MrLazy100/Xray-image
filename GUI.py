from tkinter import *
from tkinter import filedialog
import os
import numpy as np         
import time
from datetime import datetime
import shutil
import threading
import skimage
from skimage.transform import resize
import cv2                 
from keras.models import load_model
window= Tk()

class g:
    d1=''
    d2=''
    def __init__(self):
        pass
gg=g()

c_dir=__file__
c_dir=c_dir.strip(os.path.basename(__file__))
def insert_im():
    dir1 = filedialog.askdirectory()+'/'
    gg.d1=dir1
    label2.configure(text=dir1)
def valid_im():
    dir2 = filedialog.askdirectory()+'/'
    gg.d2=dir2
    label3.configure(text=dir2)
def test_im():
    im_dir=filedialog.askopenfilename(filetypes = (("png","*.png"),("all files","*.*")))
    img = cv2.imread(im_dir)
    img = skimage.transform.resize(img, (64, 64, 3))
    img = np.asarray(img)
    model = load_model(c_dir+'temp.h5')
    x=[]
    x.append(img)
    x = np.asarray(x)
    pred_prob=model.predict(x)
    if pred_prob[0]>0.6:
        pred_lbl='Positive'
    else:
        pred_lbl='Negative'
    label5.configure(text='sigmoid value:-'+str(pred_prob[0])+'||Result:-'+pred_lbl)
def start_submit_thread(event):
    global submit_thread
    submit_thread = threading.Thread(target=train_model)
    submit_thread.daemon = True
    submit_thread.start()
    window.after(20, check_submit_thread)

def check_submit_thread():
    if submit_thread.is_alive():
        window.after(20, check_submit_thread)
    else:
        pass
def train_model():
    label4.configure(text='In Progres...')
    os.system('python nn.py '+gg.d1+' '+gg.d2)
    label4.configure(text='Completed')
#def on_click():
 #   import os
  #  os.chdir("C:\Users\premp\chest_xray\Untitled2.ipynb")
   # os.system("demo_1.bat")
    
    
       
    
window.geometry("600x600")
window.title("Welcome")

        
        
label1=Label(window,text="Check for Pneumonia",fg="white",background="black",font=("ariel",16,"bold")).pack()

button1=Button(window,text="Training set images",fg="black",bg= "red",command=insert_im,relief="raised",font=("ariel",12,"bold"),width=20)
button1.place(x=400,y=50)

button2=Button(window,text="Validation set images",fg="black",bg= "red",command=valid_im,relief="raised",font=("ariel",12,"bold"),width=20)
button2.place(x=400,y=100)

button3=Button(window,text="Insert image",fg="black",bg= "red",relief="raised",command=test_im,font=("ariel",12,"bold"),width=20)
button3.place(x=400,y=250)


label2=Label(window,text="path not specified",width=40)
label2.place(x=50,y=50)

label3=Label(window,text="path not specified",width=40)
label3.place(x=50,y=100)

label4=Label(window,text="Process not started",width=40)
label4.place(x=50,y=150)

label5=Label(window,text="Image not selected",width=40)
label5.place(x=50,y=250)

button4=Button(window,text="Train now",fg= "green",bg="black",relief="groove",font=("ariel",12,"bold"),command=lambda:start_submit_thread(None))
button4.place(x=400,y=150)


window.mainloop()
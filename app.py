# PATCH 1.6: Full elements + Model applied in both tab
# To-do: Checkid button function (check if the student id existed), arrange elements for better gui

import utils as u
from face_detector import YoloV5FaceDetector
import tensorflow as tf
import tkinter as tk
from customtkinter import CTkImage
import os
import datetime
from PIL import Image, ImageTk
import cv2
import customtkinter as ctk
import csv
import numpy as np
from numpy import asarray

# Set the desired width and height
desired_width = 128
desired_height = 108

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

face_crop = YoloV5FaceDetector()
model_interf = "GhostFaceNet_W1.3_S1_ArcFace.h5"
if isinstance(model_interf, str) and model_interf.endswith(".h5"):
    model = tf.keras.models.load_model(model_interf)
    model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
else:
    model_interf = model_interf

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Attendance System")

        # ======= Window Size ======= #
        self.geometry("1280x1080")
        self.resizable(True, True)

        self.tab_view = MyTabView(master=self)

class MyTabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # create tabs
        self.tab_1 = self.add("Register")
        self.tab_2 = self.add("Check Attendance")

        self.pack(anchor="c", fill="both", expand=True, padx=30, pady=30)

        # ======== Tab 1: Register ========= #
        # 1.1/ Create a textbox in the first tab
        self.textbox = ctk.CTkEntry(master=self.tab_1, width=240, height=60, corner_radius=20, placeholder_text="Enter your Student ID:")
        self.textbox.pack()

        self.get_studentid = None
        # 1.1.1/ Check student ID
        self.check_studentid = ctk.CTkButton(master=self.tab_1, text="Check Student ID",corner_radius=20, command=self.check_name)
        self.check_studentid.pack(padx=8,pady=8)

        # 1.2/ Create frame for display capture images
        self.main_frame = ctk.CTkFrame(self.tab_1, width=20, height=30)
        self.main_frame.pack(padx=30, pady=30)

        # 1.2.1/ Create a container frame for each image
        self.container = ctk.CTkFrame(master=self.main_frame, width=120, height=120)
        self.container.pack(padx=10, pady=10)

        # 1.2.2/ Create label - image - student name
        self.label = [ctk.CTkLabel(master=self.container, text="") for _ in range(3)]
        self.img_labels = [ctk.CTkLabel(master=self.container, text="", bg_color='transparent', width=100, height=100) for _ in range(3)]
        self.studentid = [ctk.CTkLabel(master=self.container, text="") for _ in range(3)]

        for i in range(3):
            self.label[i].grid(row=0, column=i, padx=5, pady=5)
            self.img_labels[i].grid(row=1, column=i, padx=5, pady=5)
            self.studentid[i].grid(row=2, column=i, padx=5, pady=5)

        # 1.2.3/ Successfully register message
        self.regis = ctk.CTkLabel(master=self.main_frame, width=100, height=10,text="")
        self.regis.pack(padx=10, pady=10)

        # 1.3/ Webcam & Capture button
        # 1.3.1/ Capture button
        self.download_button = ctk.CTkButton(self.tab_1, text="Capture", command=self.on_button_click)
        self.download_button.pack()

        self.button_count = 0
        # 1.3.2/ Webcam
        # 1.3.2.1/ Create frame for webcam capture
        self.frame1 = ctk.CTkFrame(self.tab_1, width=0, height=0)
        self.frame1.pack(padx=30, pady=30)

        # 1.3.2.2/ Create camera frame
        self.cameraFrame1 = ctk.CTkFrame(master=self.frame1, width=0)
        self.cameraFrame1.pack(expand=True, fill="both")

        self.camera1 = ctk.CTkLabel(self.cameraFrame1, text="")
        self.camera1.pack()

        self.img = None
        self.img1 = None
        self.img2 = None
        self.img3 = None

        # ======== Tab 2: Check Attendance ========= #
        # 2.1/
        self.frame2 = ctk.CTkFrame(self.tab_2, width=0, height=0)
        self.frame2.pack(padx=30, pady=30)

        # 1.3.2.2/ Create camera frame
        self.cameraFrame2 = ctk.CTkFrame(master=self.frame2, width=0)
        self.cameraFrame2.pack(expand=True, fill="both")

        self.camera2 = ctk.CTkLabel(self.cameraFrame2, text="")
        self.camera2.pack()

        # 2.2/
        self.checkin_button = ctk.CTkButton(self.tab_2, text="Capture", command=self.checkin_button_click)
        self.checkin_button.pack()

        self.checkin_button_count = 0
        self.checkin_img = None

        self.checkin_main_frame = ctk.CTkFrame(self.tab_2, width=20, height=30)
        self.checkin_main_frame.pack(padx=30, pady=30)

        # 1.2.1/ Create a container frame for each image
        self.checkin_container = ctk.CTkFrame(master=self.checkin_main_frame, width=120, height=120)
        self.checkin_container.pack(padx=10, pady=10)

        # 1.2.2/ Create image
        self.checkin_img_labels = ctk.CTkLabel(master=self.checkin_container, text="", bg_color='transparent', width=100, height=100)
        self.checkin_img_labels.grid(row=0, column=0, padx=5, pady=5)

        # 1.2.3/ Successfully register message
        self.checkin_label = ctk.CTkLabel(master=self.checkin_main_frame, width=100, height=10,text="")
        self.checkin_label.pack(padx=10, pady=10)

    # ==== tab2: METHODS ==== #
    def checkin_button_click(self):
        self.checkin_button_count += 1

        if self.checkin_button_count == 1:
            self.display_checkin_captured_photo()
            self.checkin_img = self.checkin_image_to_array(self.checkin_img_labels.image)
            self.checkin(self.checkin_img)
            self.checkin_button_count = 0


    def display_checkin_captured_photo(self):
        checkin_img = self.checkin_img.resize((128, 72))
        checkin_img_tk = ImageTk.PhotoImage(checkin_img)
        self.checkin_img_labels.configure(image=checkin_img_tk)
        self.checkin_img_labels.image = checkin_img_tk

    def checkin(self, img):
        img = u.resize_image(img)
        image_id,image_list,distances,person_id  = u.return_id_imgs(img,model_interf,face_crop)
        person_name = u.find_string_by_person_id(person_id)
        string=f"Checkin: {person_name}!😎"
        self.checkin_label.configure(text=string)
        self.checkin_label._text=string

    def checkin_image_to_array(self, photoimage):
        # Convert PIL Image to NumPy array
        checkin_numpydata = asarray(self.checkin_img)
        return checkin_numpydata

    # ==== tab1: METHODS ==== #
    def check_name(self):
        studentid = self.textbox.get()
        self.get_studentid = studentid

        return self.get_studentid

    def get_max_user_id(self):
        max_id = 999
        if os.path.isfile("./static/data.csv"):
            with open("./static/data.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:  # Check if the row is not empty
                        user_id = int(row[0])
                        if user_id > max_id:
                            max_id = user_id
        return max_id

    def add_new_person(self, name, img1, img2, img3):
        nid = self.get_max_user_id() + 1
        imgs = [img1, img2, img3]
        imgs = [u.resize_image(imgg) for imgg in imgs]
        print(len(imgs))
        print("new ID: ", nid)
        a, _, _ = u.add_new_person(imgs, nid, model_interf, face_crop, name)
        if (a.shape[0] - 1 != int(nid)):
            print('ERROR: Something goes wrong with adding this new person, current embedding shape: ' + str(a.shape))
        else:
            return self.regis_mess()

    def streaming(self):
        _, img = cap.read()
        _, checkin_img = cap.read()
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        checkin_cv2image = cv2.cvtColor(checkin_img, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(cv2image)
        self.checkin_img = Image.fromarray(checkin_cv2image)
        ImgTks = ImageTk.PhotoImage(image=self.img)
        checkin_ImgTks = ImageTk.PhotoImage(image=self.checkin_img)

        self.camera1.imgtk = ImgTks
        self.camera1.configure(image=ImgTks)

        self.camera2.imgtk = checkin_ImgTks
        self.camera2.configure(image=checkin_ImgTks)

        self.after(20, self.streaming)

    def image_to_array(self, photoimage):
        # Convert PIL Image to NumPy array
        numpydata = asarray(self.img)
        return numpydata

    def display_captured_photo(self, index):
        label = ["image 1", "image 2", "image 3"]
        img = self.img.resize((128, 72))
        img_tk = ImageTk.PhotoImage(img)
        self.label[index].configure(text=label[index])
        self.label[index]._text = label
        self.studentid[index].configure(text=self.get_studentid)
        self.studentid[index]._text = self.get_studentid
        self.img_labels[index].configure(image=img_tk)
        self.img_labels[index].image = img_tk

    def regis_mess(self):
        string="Successfully register your attendance!😎"
        self.regis.configure(text=string)
        self.regis._text=string

    def on_button_click(self):
        self.button_count += 1

        if 1 <= self.button_count <= 3:
            self.display_captured_photo(self.button_count - 1)

        if self.button_count == 1:
            self.img1 = self.image_to_array(self.img_labels[self.button_count - 1].image)
        if self.button_count == 2:
            self.img2 = self.image_to_array(self.img_labels[self.button_count - 1].image)
        if self.button_count == 3:
            name = self.get_studentid
            self.img3 = self.image_to_array(self.img_labels[self.button_count - 1].image)
            self.add_new_person(name, self.img1, self.img2, self.img3)
            self.button_count = 0

if __name__ == "__main__":
    app = App()
    app.tab_view.streaming()
    app.mainloop()


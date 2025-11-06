# import  tkinter library
from tkinter import *
# access messagebox and ttk's function (ttk -> combobox)
from tkinter import ttk,messagebox
# access filedialogue box 
from tkinter import filedialog
# import pillow library for images
from PIL import  ImageTk, Image
from datetime import date
import sys
import os
# import pyglet
import threading


# global variables
input_dir = None
input_image_for_evaluation = None

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS2
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path,relative_path)


def dashboard_fn(loginscreen_image,mainframe,Width,Height):

    #design_1
    design_1 = Label(mainframe,bd=2,bg='black')
    design_1.place(x=Width*0.025,y=Height*0.135,width=Width*0.93,height=Height*0.003)

    #design_2
    design_2 = Label(mainframe,bd=2,bg='black')
    design_2.place(x=Width*0.025,y=Height*0.2,width=Width*0.93,height=Height*0.003)

    # --------------------------------------------------------------------------------------------------
    # row 1
    # --------------------------------------------------------------------------------------------------

    # logo image (use PIL library) -> access image from the system
    logo_image = Image.open(resource_path("Images\\logo.png"))
    # resize the image and apply a high-quality down sampling filter
    logo_image = logo_image.resize((int(Height*0.23),int(Height*0.1)), Image.Resampling.LANCZOS)
    # PhotoImage class is used to add image to widgets, icons etc
    logo_image = ImageTk.PhotoImage(logo_image)

    #logo_label
    logo_label = Label(mainframe,bd=0,bg='#f0f0f0',image=logo_image)
    logo_label.image = logo_image
    logo_label.place(x=Width*0.025,y=Height*0.02,width=Width*0.12,height=Height*0.1)

    # # upload_dataset_label
    # font_  = ('Lexend','9','bold')
    # upload_dataset_label = Button(mainframe,text='Upload Dataset :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w', cursor='hand2')
    # upload_dataset_label.place(x=Width*0.025,y=Height*0.154,width=Width*0.08,height=Height*0.03)

    # # path_label
    # font_  = ('Lexend','9','bold')
    # path_label = Button(mainframe,text='Dataset Path :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w', cursor='hand2')
    # path_label.place(x=Width*0.188,y=Height*0.154,width=Width*0.07,height=Height*0.03)

    # #path_entry
    # path_entry = Entry(mainframe,justify=LEFT,bd=1,fg='black',bg='#f0f0f0',font=('Lexend','9'), state='readonly')
    # path_entry.place(x=Width*0.27,y=Height*0.148,width=Width*0.31,height=Height*0.04)
    # path_entry.insert(0,'    ')
    
    # # dataset_status_label
    # font_  = ('Lexend','9','bold')
    # dataset_status_label = Button(mainframe,text='Dataset Status :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w', cursor='hand2')
    # dataset_status_label.place(x=Width*0.59,y=Height*0.154,width=Width*0.1,height=Height*0.03)

    # #dataset_status_entry
    # dataset_status_entry = Entry(mainframe,justify=LEFT,bd=1,fg='black',bg='#f0f0f0',font=('Lexend','9'), state='readonly')
    # dataset_status_entry.place(x=Width*0.675,y=Height*0.148,width=Width*0.28,height=Height*0.04)
    # dataset_status_entry.insert(0,'   ')

    # --------------------------------------------------------------------------------------------------
    # row 2
    # --------------------------------------------------------------------------------------------------s

    # frame 2
    # training_performance_measures_frame
    font_  = ('Lexend','9','bold') #A3BFD9
    training_performance_measures_frame = LabelFrame(mainframe,bg='#f0f0f0',bd=2,font=font_,text='  Training Performance Measures  ')
    training_performance_measures_frame.place(x=Width*0.275,y=Height*0.21,width=Width*0.43,height=Height*0.7)

    # 01
    # accuracy_label
    font_  = ('Lexend','9','bold')
    accuracy_label = Label(training_performance_measures_frame,text='Accuracy :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    accuracy_label.place(x=Width*0.01,y=Height*0.024,width=Width*0.19,height=Height*0.03)

    #accuracy_label_design
    accuracy_label_design = Label(training_performance_measures_frame,bd=2,bg='black')
    accuracy_label_design.place(x=Width*0.01,y=Height*0.057,width=Width*0.19,height=Height*0.003)

    #accuracy_entry
    accuracy_entry = Entry(training_performance_measures_frame,justify=LEFT,bd=0,fg='black',bg='#f0f0f0',font=('Lexend','18'), state='readonly')
    accuracy_entry.place(x=Width*0.01,y=Height*0.06,width=Width*0.19,height=Height*0.05)
    accuracy_entry.insert(0,'0.00000')

    # 02
    # precision_label
    font_  = ('Lexend','9','bold')
    precision_label = Label(training_performance_measures_frame,text='Precision :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    precision_label.place(x=Width*0.225,y=Height*0.024,width=Width*0.19,height=Height*0.03)

    #precision_label_design
    precision_label_design = Label(training_performance_measures_frame,bd=2,bg='black')
    precision_label_design.place(x=Width*0.225,y=Height*0.057,width=Width*0.19,height=Height*0.003)

    #precision_entry
    precision_entry = Entry(training_performance_measures_frame,justify=LEFT,bd=0,fg='black',bg='#f0f0f0',font=('Lexend','18'), state='readonly')
    precision_entry.place(x=Width*0.225,y=Height*0.06,width=Width*0.19,height=Height*0.05)
    precision_entry.insert(0,'0.00000')

    # 03
    # recall_label
    font_  = ('Lexend','9','bold')
    recall_label = Label(training_performance_measures_frame,text='Recall :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    recall_label.place(x=Width*0.01,y=Height*0.13,width=Width*0.19,height=Height*0.03)

    #recall_label_design
    recall_label_design = Label(training_performance_measures_frame,bd=2,bg='black')
    recall_label_design.place(x=Width*0.01,y=Height*0.163,width=Width*0.19,height=Height*0.003)

    #recall_entry
    recall_entry = Entry(training_performance_measures_frame,justify=LEFT,bd=0,fg='black',bg='#f0f0f0',font=('Lexend','18'), state='readonly')
    recall_entry.place(x=Width*0.01,y=Height*0.166,width=Width*0.19,height=Height*0.05)
    recall_entry.insert(0,'0.00000')

    # 04
    # f1_measure_label
    font_  = ('Lexend','9','bold')
    f1_measure_label = Label(training_performance_measures_frame,text='F1 Measure :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    f1_measure_label.place(x=Width*0.225,y=Height*0.13,width=Width*0.19,height=Height*0.03)

    #f1_measure_label_design
    f1_measure_label_design = Label(training_performance_measures_frame,bd=2,bg='black')
    f1_measure_label_design.place(x=Width*0.225,y=Height*0.163,width=Width*0.19,height=Height*0.003)

    #f1_measure_entry
    f1_measure_entry = Entry(training_performance_measures_frame, justify=LEFT, bd=0, fg='black', bg='#f0f0f0', font=('Lexend', '18'), state='readonly')
    f1_measure_entry.place(x=Width*0.225, y=Height*0.166, width=Width*0.19, height=Height*0.05)
    f1_measure_entry.insert(0, '96')
    # 04
    # confusin_matrix_label
    font_  = ('Lexend','9','bold')
    confusin_matrix_label = Label(training_performance_measures_frame,text='Confusion Matrix :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    confusin_matrix_label.place(x=Width*0.01,y=Height*0.23,width=Width*0.19,height=Height*0.03)

    #confusin_matrix_label_design
    confusin_matrix_label_design = Label(training_performance_measures_frame,bd=2,bg='black')
    confusin_matrix_label_design.place(x=Width*0.01,y=Height*0.263,width=Width*0.405,height=Height*0.003)

    # confusion_matrix_entry_label
    confusion_matrix_entry_label = Label(training_performance_measures_frame,fg='white',text='Confusion Matrix Image',bg='black',bd=0)
    confusion_matrix_entry_label.place(x=Width*0.01,y=Height*0.28,width=Width*0.405,height=Height*0.375)
    img = Image.open(os.path.join('assets','confusion_matrix.jpg'))
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((int(Width*0.45),int(Width*0.215)), Image.Resampling.LANCZOS)
    # PhotoImage class is used to add image to widgets, icons etc
    final_image = ImageTk.PhotoImage(img)

    # Set the image in the label
    confusion_matrix_entry_label.config(image=final_image)
    confusion_matrix_entry_label.image = final_image

    accuracy_entry.config(state='normal')
    precision_entry.config(state='normal')
    recall_entry.config(state='normal')
    f1_measure_entry.config(state='normal')
    
    accuracy_entry.delete(0, END)
    precision_entry.delete(0, END)
    recall_entry.delete(0, END)
    f1_measure_entry.delete(0, END)
    #accuracy = 2 * precision * recall / (precision + recall)
    accuracy_entry.insert(0, "78.33%")
    precision_entry.insert(0, "74.3%")
    recall_entry.insert(0, "82.2%")
    f1_measure_entry.insert(0, "75%")

    accuracy_entry.config(state='readonly')
    precision_entry.config(state='readonly')
    recall_entry.config(state='readonly')
    f1_measure_entry.config(state='readonly')

    # frame 3
    # evaluation_performance_measures_frame
    font_  = ('Lexend','9','bold')
    evaluation_performance_measures_frame = LabelFrame(mainframe,bg='#f0f0f0',bd=2,font=font_,text='  Evaluation Performance Measures  ')
    evaluation_performance_measures_frame.place(x=Width*0.715,y=Height*0.21,width=Width*0.24,height=Height*0.7)

    # upload_image_label
    font_  = ('Lexend','9','bold')
    upload_image_label = Label(evaluation_performance_measures_frame,text='Upload MRI/CT Scan Image :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    upload_image_label.place(x=Width*0.01,y=Height*0.034,width=Width*0.14,height=Height*0.03)


    #upload_image_status_entry
    upload_image_status_entry = Entry(evaluation_performance_measures_frame,justify=RIGHT,bd=0,fg='red',bg='#f0f0f0',font=('Lexend','8'), state='readonly')
    upload_image_status_entry.place(x=Width*0.01,y=Height*0.075,width=Width*0.215,height=Height*0.04)
    upload_image_status_entry.insert(0,'No MRI CT Scan Image Found!')

    # image_entry [image will be shown within this label]
    image_entry = Label(evaluation_performance_measures_frame,bg='white',bd=0)
    image_entry.place(x=Width*0.01,y=Height*0.125,width=Width*0.215,height=Height*0.28)

    # 01
    # evaluation_status_label
    font_  = ('Lexend','9','bold')
    evaluation_status_label = Label(evaluation_performance_measures_frame,text='Tumor Status :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    evaluation_status_label.place(x=Width*0.01,y=Height*0.47,width=Width*0.19,height=Height*0.03)

    #evaluation_status_label_design
    evaluation_status_label_design = Label(evaluation_performance_measures_frame,bd=2,bg='black')
    evaluation_status_label_design.place(x=Width*0.01,y=Height*0.503,width=Width*0.215,height=Height*0.003)

    #evaluation_status_entry
    evaluation_status_entry = Entry(evaluation_performance_measures_frame,justify=LEFT,bd=0,fg='red',bg='#f0f0f0',font=('Lexend','14'), state='readonly')
    evaluation_status_entry.place(x=Width*0.01,y=Height*0.506,width=Width*0.215,height=Height*0.05)
    evaluation_status_entry.insert(0,'Tumor Found!')

    # 02
    # evaluation_accuracy_label
    font_  = ('Lexend','9','bold')
    evaluation_accuracy_label = Label(evaluation_performance_measures_frame,text='Evaluation Accuracy :',font=font_,fg='black',bg='#f0f0f0',bd=0,justify=LEFT,anchor='w')
    evaluation_accuracy_label.place(x=Width*0.01,y=Height*0.56,width=Width*0.19,height=Height*0.03)

    #evaluation_accuracy_label_design
    evaluation_accuracy_label_design = Label(evaluation_performance_measures_frame,bd=2,bg='black')
    evaluation_accuracy_label_design.place(x=Width*0.01,y=Height*0.593,width=Width*0.215,height=Height*0.003)

    #evaluation_accuracy_label_entry
    evaluation_accuracy_label_entry = Entry(evaluation_performance_measures_frame,justify=LEFT,bd=0,fg='black',bg='#f0f0f0',font=('Lexend','18'), state='readonly')
    evaluation_accuracy_label_entry.place(x=Width*0.01,y=Height*0.596,width=Width*0.215,height=Height*0.05)
    evaluation_accuracy_label_entry.insert(0,'0.00000')





    # FUNCTIONS
    #upload_image_fn
    def upload_image_fn():
        global input_image_for_evaluation
        # Open file dialog box and get the image file path
        input_image_for_evaluation = os.path.normpath(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]))
        if input_image_for_evaluation:
            # loginscreen_image (use PIL library) -> access image from the system
            img = Image.open(input_image_for_evaluation)
            # resize the image and apply a high-quality down sampling filter
            img = img.resize((int(Width*0.14),int(Width*0.14)), Image.Resampling.LANCZOS)
            # PhotoImage class is used to add image to widgets, icons etc
            final_image = ImageTk.PhotoImage(img)

            # Set the image in the label
            image_entry.config(image=final_image)
            image_entry.image = final_image

            upload_image_status_entry.delete(0,END)
            # upload_image_status_entry.insert(0,'MRI CT Scan Uploaded!')

    #upload_image_button
    font_  = ('Lexend','9','bold')
    upload_image_button = Button(evaluation_performance_measures_frame,text='Upload',font=font_,fg='black',bg='#A3BFD9',bd=0,justify=CENTER,
                             command=upload_image_fn,activebackground='white',activeforeground='black', cursor='hand2')
    upload_image_button.place(x=Width*0.162,y=Height*0.028,width=Width*0.06,height=Height*0.04)

    def start_evaluate_tumor():
        thread = threading.Thread(target=evaluate_tumor_thread)
        thread.start()
        
    def evaluate_tumor_thread():
        global input_dir
        global input_image_for_evaluation
        
        # Dataset selection validation
        # Check for model file
        model_path = os.path.join(os.path.join('model'), 'best.pt')
        if not os.path.exists(model_path):
            show_error('Model is not available!\n Please do model training first!')
            return

        # Image selection validation
        if not input_image_for_evaluation:
            show_error('Image is not selected!')
            return
        
        evaluation_status_entry.config(state='normal')
        evaluation_accuracy_label_entry.config(state='normal')
        
        evaluation_status_entry.delete(0, END)
        evaluation_accuracy_label_entry.delete(0, END)
        
        evaluation_status_entry.insert(0, 'Computing...')
        evaluation_accuracy_label_entry.insert(0, 'Computing...')
    
        from predict import predict_tumor
        result, accuracy_confidence = predict_tumor(input_image_for_evaluation, model_path)
        img = Image.open(os.path.join("yolov5_local/runs/detect/exp/check.jpg"))
        # resize the image and apply a high-quality down sampling filter
        img = img.resize((int(Width*0.14),int(Width*0.14)), Image.Resampling.LANCZOS)
        # PhotoImage class is used to add image to widgets, icons etc
        final_image = ImageTk.PhotoImage(img)
        # Set the image in the label
        image_entry.config(image=final_image)
        image_entry.image = final_image
        if result: 
            tumors = {"0": "No Tumor","1": "Benign Tumor","2": "Malignant Tumor"}
            result = tumors[result]
        else:
             result = "No detection"
        if result and accuracy_confidence:
            
            evaluation_status_entry.delete(0, END)
            evaluation_accuracy_label_entry.delete(0, END)
            
            evaluation_status_entry.insert(0, f"{result}")
            evaluation_accuracy_label_entry.insert(0, f"{accuracy_confidence:.5f}")
            
            evaluation_status_entry.config(state='readonly')
            evaluation_accuracy_label_entry.config(state='readonly')
        else:
            evaluation_status_entry.delete(0, END)
            evaluation_accuracy_label_entry.delete(0, END)
            
            evaluation_status_entry.insert(0, 'Failed!')
            evaluation_accuracy_label_entry.insert(0, 'Failed!')
            
            evaluation_status_entry.config(state='readonly')
            evaluation_accuracy_label_entry.config(state='readonly')
        
    #evaluate_button
    font_  = ('Lexend','9','bold')
    evaluate_button = Button(evaluation_performance_measures_frame,text='Evaluate Tumor',font=font_,fg='black',bg='#A3BFD9',bd=0,justify=CENTER,
                             command=start_evaluate_tumor,activebackground='white',activeforeground='black', cursor='hand2')
    evaluate_button.place(x=Width*0.01,y=Height*0.42,width=Width*0.215,height=Height*0.04)
 

    # RESET FUNCTION
    def reset_fn_():
        
        upload_image_status_entry.config(state='normal')
        evaluation_status_entry.config(state='normal')
        evaluation_accuracy_label_entry.config(state='normal')
        
        
        upload_image_status_entry.delete(0,END)
        evaluation_status_entry.delete(0,END)
        evaluation_accuracy_label_entry.delete(0,END)
        
        evaluation_status_entry.config(state='readonly')
        evaluation_accuracy_label_entry.config(state='readonly')

        # remove image
        image_entry.config(image='')
        image_entry.image = None
        upload_image_status_entry.insert(0,'No MRI CT Scan Image Found!')
        
    #reset_button
    font_  = ('Lexend','12','bold')
    reset_button = Button(mainframe,text='Reset',font=font_,fg='black',bg='#A3BFD9',bd=0,justify=CENTER,
                             command=reset_fn_,activebackground='white',activeforeground='black', cursor='hand2')
    reset_button.place(x=Width*0.745,y=Height*0.033,width=Width*0.1,height=Height*0.065)


    # LOGOUT FUNCTION
    def logout_fn_():
        # clear the previous frame
        for widgets in mainframe.winfo_children():
            widgets.destroy()

        from login import login_fn
        login_fn(loginscreen_image,mainframe,Width,Height)

    #logout_button
    font_  = ('Lexend','12','bold')
    logout_button = Button(mainframe,text='Logout',font=font_,fg='black',bg='#A3BFD9',bd=0,justify=CENTER,
                             command=logout_fn_,activebackground='white',activeforeground='black', cursor='hand2')
    logout_button.place(x=Width*0.855,y=Height*0.033,width=Width*0.1,height=Height*0.065)
    
    
# General Methods
def show_warning(msg):
    messagebox.showwarning("Warning", msg)
    
def show_error(msg):
    messagebox.showerror("Error", msg)
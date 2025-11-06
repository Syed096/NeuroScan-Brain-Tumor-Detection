# import  tkinter library
from tkinter import *
# access messagebox and ttk's function (ttk -> combobox)
from tkinter import ttk,messagebox
# access filedialogue box 
from tkinter import filedialog as fd
# import pillow library for images
from PIL import  ImageTk, Image

def login_fn(loginscreen_image,mainframe,Width,Height):        

    #login_image_label
    login_image_label = Label(mainframe,image=loginscreen_image,bd=0)
    login_image_label.place(x=0,y=0,height=Height*0.928,width=Width*0.977)

    #username_entry
    username_entry = Entry(mainframe,justify=LEFT,bd=0,fg='black',bg='white',font=('Lexend','12'))
    username_entry.place(x=Width*0.647,y=Height*0.405,width=Width*0.23,height=Height*0.038)
    username_entry.insert(0,'amina')

    #password_entry
    password_entry = Entry(mainframe,justify=LEFT,show='*',bd=0,fg='black',bg='white',font=('Lexend','12'))
    password_entry.place(x=Width*0.647,y=Height*0.505,width=Width*0.23,height=Height*0.038)
    password_entry.insert(0,'amina123')

    #login_fn
    def login_fn_():
        # access entries
        username = username_entry.get()
        password = password_entry.get()

        if (username_entry.get()=="" or password_entry.get() == ""):
            messagebox.showerror("ERROR","Fields Should Not Be Empty !",parent=mainframe)
        else:
            if username != 'amina' or password != 'amina123':
                messagebox.showerror("Error","Invalid  Gmail Or Password",parent=mainframe)
            else:
                messagebox.showinfo("Congrats","Log-in Successful",parent=mainframe)
                # clear the previous frame
                for widgets in mainframe.winfo_children():
                    widgets.destroy()

                # calling the dashboard
                from dashboard import dashboard_fn
                dashboard_fn(loginscreen_image,mainframe,Width,Height)
                username_entry.delete(0,END)
                password_entry.delete(0,END)

        print('Successfully Login')

    #login_button
    font_  = ('Lexend','16','bold')
    login_button = Button(mainframe,text='Login',font=font_,fg='black',bg='white',bd=0,justify=CENTER,
                             command=login_fn_,activebackground='white',activeforeground='black', cursor='hand2')
    login_button.place(x=Width*0.545,y=Height*0.64,width=Width*0.33,height=Height*0.065)
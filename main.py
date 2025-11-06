# -----------------------------------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------------------------------

# import  tkinter library
from tkinter import *
# access messagebox and ttk's function (ttk -> combobox)
from tkinter import ttk,messagebox
# access filedialogue box 
from tkinter import filedialog as fd
# import pillow library for images
from PIL import  ImageTk, Image

# -----------------------------------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------------------------------

def main_():
    # -----------------------------------------------------------------------------------------------------
    # page setting
    # -----------------------------------------------------------------------------------------------------

    main_page = Tk()
    Width = main_page.winfo_screenwidth()
    Height = main_page.winfo_screenheight()
    main_page.geometry('%dx%d+0+0' % (Width,Height)) 
    # background color
    main_page.configure(background="#f0f0f0") 
    # title of page
    main_page.title("Brain Tumor Detection")
    # vertically , horizontally resizing accessibility
    main_page.resizable(False, False) 
    main_page.state('zoomed')
    main_page.attributes('-topmost', True)

    # loginscreen_image (use PIL library) -> access image from the system
    loginscreen_image = Image.open("Images\Login.png")
    # resize the image and apply a high-quality down sampling filter
    loginscreen_image = loginscreen_image.resize((int(Width*0.98),int(Height*0.935)), Image.Resampling.LANCZOS)
    # PhotoImage class is used to add image to widgets, icons etc
    loginscreen_image = ImageTk.PhotoImage(loginscreen_image)

    # -----------------------------------------------------------------------------------------------------
    # mainframe
    # -----------------------------------------------------------------------------------------------------
    mainframe = LabelFrame(main_page,bg='#f0f0f0',bd=2)
    mainframe.place(x=Width*0.01,y=Height*0.02,width=Width*0.98,height=Height*0.935)

    from login import login_fn
    login_fn(loginscreen_image,mainframe,Width,Height)

    main_page.mainloop()
    
# calling main
main_()

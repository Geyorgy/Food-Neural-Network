from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

food_list = ['chicken_wings', 'cup_cakes', 'donuts', 'dumplings',
             'french_fries', 'fried_rice', 'greek_salad', 'hamburger',
             'hot_dog', 'ice_cream', 'lasagna', 'macaroni_and_cheese',
             'omelette', 'pancakes', 'pizza', 'ramen', 'spaghetti_bolognese',
             'steak', 'sushi', 'waffles']


def predict_class(model, img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    return food_list[index]


window = Tk()
window.title("Food Neural Network")
window.geometry('800x540')


def button_click():
    filename = filedialog.askopenfilename(initialdir="/", title="Choose a file", filetypes=[('picture files', '*.jpg')])
    file = str(filename)
    lst.insert(END, file)


def show_image(e):
    n = lst.curselection()
    img = Image.open(lst.get(n))
    img_forNN = img.resize((300, 300))
    img = img.resize((500, 300))
    canvas.image = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=canvas.image, anchor=NW)
    labelNNAnswerInfo = Label(window, text='Ответ нейросети: ').place(x=350, y=430)
    model_best = load_model('best_model20.hdf5', compile=False)
    labelNNAnswer = Label(window, text='                         ').place(x=457, y=430)
    labelNNAnswer = Label(window, text=predict_class(model_best, img_forNN)).place(x=457, y=430)


lst = Listbox(window, width=40)
lst.pack(side='left', fill=Y, expand=1)
lst.bind("<<ListboxSelect>>", show_image)

canvas = Canvas(window, width=500, height=300, bg='white')
canvas.pack(side='left')

btnBrows = Button(window, text="Browse", width=30, command=button_click).place(x=440, y=43)

window.mainloop()
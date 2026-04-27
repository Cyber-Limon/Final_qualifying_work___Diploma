import joblib
import pandas as pd

from tkinter    import *
from tensorflow import keras





compound   = pd.read_excel("ВКР, датасет.xlsx", nrows=0, usecols="B:V")
compound   = compound.columns.tolist()

properties = pd.read_excel("ВКР, датасет.xlsx", nrows=0, usecols="W:AD")
properties = properties.columns.tolist()



scaler_X = joblib.load("ВКР, нормализаторы\\scaler_X.pkl")
scaler_Y = joblib.load("ВКР, нормализаторы\\scaler_Y.pkl")



models = []
for i in range(8):
    model = keras.models.load_model(f"ВКР, модели\\ВКР, модель {i + 1}.keras")
    models.append(model)





def clean_compound():
    clean_properties()

    for entry in entries_compound:
        entry.delete(0, END)



def clean_properties():
    title_3.config(text="")

    for entry in entries_properties:
        entry.config(state='normal')
        entry.delete(0, END)
        entry.config(state='readonly')



def on_compound_change(event):
    title_3.config(text="")

    clean_properties()



def forecast():
    current_compound = []
    for entry in entries_compound:
        current_compound.append(float(entry.get()) if entry.get() != '' else 0)

    if sum(current_compound) != 100:
        title_3.config(text=f"ПРЕДУПРЕЖДЕНИЕ: неправильный ввод - сумма элементов состава ({sum(current_compound)}) не равна 100")
        return
    else:
        title_3.config(text="")

    current_compound   = scaler_X.transform([current_compound])

    current_properties = []
    for model in models:
        predict = model.predict(current_compound, verbose=0)
        current_properties.append(predict[0][0])

    current_properties = scaler_Y.inverse_transform([current_properties])

    for i in range(len(entries_properties)):
        entries_properties[i].config(state='normal')
        entries_properties[i].delete(0, END)
        entries_properties[i].insert(0, round(current_properties[0][i], 5))
        entries_properties[i].config(state='readonly')



def validate_input(new_value):
    if new_value == "":
        return True

    try:
        float(new_value)
        return True
    except ValueError:
        return False





root = Tk()
root.title("Прогнозирование свойств")



columns = 16
font = ('Calibri Bold', 20)

vcmd = (root.register(validate_input), '%P')



title_1 = Label(root, text="ПРОГНОЗИРОВАНИЕ СВОЙСТВ", font=('Calibri Bold', 25))
title_1.grid(row=0, column=0, columnspan=columns, pady=5)

title_2 = Label(root, text="Введите процентный состав", font=font)
title_2.grid(row=2, column=0, columnspan=columns, pady=(30, 0))

entries_compound = []
for i in range(21):
    r = i // 8 + 3
    c = (i * 2) % columns

    label = Label(root, width=5, text=compound[i], font=font)
    label.grid(row=r, column=c, pady=10)

    entry = Entry(root, width=5, font=font, justify='center', validate='key', validatecommand=vcmd)
    entry.grid(row=r, column=(c + 1), padx=10)
    entry.bind('<Key>', on_compound_change)
    entries_compound.append(entry)

button_clean = Button(root, text="Очистить состав", font=font, cursor='hand2', bg='#ff4444', fg='white', command=clean_compound)
button_clean.grid(row=5, column=10, columnspan=3, sticky='ew', padx=10)

button_forecast = Button(root, text="Выполнить прогноз", font=font, cursor='hand2', bg='#44ff44', fg='white', command=forecast)
button_forecast.grid(row=5, column=13, columnspan=3, sticky='ew', padx=10)

title_3 = Label(root, text="", font=font, fg="red")
title_3.grid(row=6, column=0, columnspan=columns)

title_4 = Label(root, text="Прогноз", font=font)
title_4.grid(row=7, column=0, columnspan=columns)

entries_properties = []
for i in range(8):
    r = i // 4 + 8
    c = (i * 4) % columns

    label = Label(root, width=10, text=properties[i], font=font)
    label.grid(row=r, column=c, columnspan=2, pady=10)

    entry = Entry(root, width=10, font=font, justify='center', state='readonly')
    entry.grid(row=r, column=(c + 2), columnspan=2, padx=10)
    entries_properties.append(entry)



root.mainloop()

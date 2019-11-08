from tkinter import *

hisse_adedi = range(1, 10)
ci = [.68, .95, .997, 1]


def show_info():
    message_txt.delete(0.0, 'end')
    selected_name = Hisse_sayisi_entry.get()
    age_restriction = confidence_interval_choice.get()

    for label in rootWindow.grid_slaves():
        if int(label.grid_info()["row"]) == 6:
            label.grid_forget()

    if age_restriction == 1:
        if selected_name == 68:
            print('confidence interval {} olarak belirlendi'.format(age_restriction))
            # gru_photo_label.grid(row=6, columnspan=3)
        elif selected_name == 95:
            print('confidence interval {} olarak belirlendi'.format(age_restriction))
            # bob_photo_label.grid(row=6, columnspan=3)
        elif selected_name == 99.7:
            print('confidence interval {} olarak belirlendi'.format(age_restriction))
            # kevin_photo_label.grid(row=6, columnspan=3)
    else:
        message_txt.insert(0.0, "You are not authorized!{}".format(selected_name))

        # restricted_photo_label.grid(row=6, columnspan=3)


rootWindow = Tk()

rootWindow.resizable(height=None, width=None)
rootWindow.title("VaR Calculation")

welcome_label = Label(rootWindow, text="Welcome to Tkinter Var Calculation", fg="gray",
                      font=("arial", 16, "bold"))

#hisse sayisi
Hisse_sayisi_entry = IntVar()
Hisse_sayisi_entry.set(1)  # default value

Hisse_sayisi = Label(rootWindow, text="Hisse:", font=("arial", 12))
dropdown = OptionMenu(rootWindow, Hisse_sayisi_entry, *hisse_adedi)

# confidence interval
confidence_interval_label = Label(rootWindow, text="Confidence Interval :", font=("arial", 12))

confidence_interval_choice = IntVar()
radio_button_one_sigma = Radiobutton(rootWindow, text="68", variable=confidence_interval_choice,
                                     font=("arial", 12), value=68)
radio_button_two_sigma = Radiobutton(rootWindow, text="95", variable=confidence_interval_choice,
                                     font=("arial", 12), value=95)
radio_button_three_sigma = Radiobutton(rootWindow, text="99.7", variable=confidence_interval_choice,
                                       font=("arial", 12), value=99.7)



message_txt = Text(rootWindow, width=16, height=1, wrap=WORD, fg="red",
                   font=("arial", 20, "bold"))

lets_see_button = Button(rootWindow, text="Hesapla", font=("arial", 12),
                         command=lambda: show_info())


# Setting the layout

welcome_label.grid(row=0, columnspan=3)
Hisse_sayisi.grid(row=1, sticky=E)
confidence_interval_label.grid(row=2, sticky=E)
dropdown.grid(row=1, column=1, sticky=E)

radio_button_one_sigma.grid(row=2, column=1, sticky=W)
radio_button_two_sigma.grid(row=2, column=2, sticky=W)
radio_button_three_sigma.grid(row=2, column=3, sticky=W)

rootWindow.grid_rowconfigure(3, minsize=20)

message_txt.grid(row=5, columnspan=3, sticky=N, rowspan=1)

lets_see_button.grid(row=4, columnspan=3)

rootWindow.mainloop()

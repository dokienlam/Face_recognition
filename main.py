import tkinter as tk
import face_input
import face_recog 

root = tk.Tk()

label = tk.Label(root, text="Name")
label.pack()

name_entry = tk.Entry(root)
name_entry.pack()

select_button = tk.Button(root, text="Input data", command = lambda: face_input.Input_data(name_entry))
select_button.pack(pady=20)

detect_button = tk.Button(root, text="Detect Faces", command = face_recog.face_recog)
detect_button.pack(pady=10)

root.mainloop()


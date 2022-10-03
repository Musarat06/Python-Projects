# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 22:17:42 2022

@author: User
"""
from tkinter import *
import numpy as np
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import statistics
from matplotlib.backend_bases import MouseButton

evt2cvs = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
if __name__ == "__main__":
    rt = Tk()
    #To set the canvas
    FM = Frame(rt, bd=3, relief=SUNKEN)
    FM.grid_rowconfigure(0, weight=1)
    FM.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(FM, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(FM,orient = VERTICAL)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(FM, bd=2, xscrollcommand=5, yscrollcommand=5)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    FM.pack(fill=BOTH,expand=1)
    
    #this fuction will be call after clicking mouse.
    X= []
    Y = []
    print("Please Enter the Coordinate location at the window by clicking using mouse.")
    print("Click cross to see the fitted lines")

    def printcoords(event): # Funciton called when mouse clicked.
        cx, cy = evt2cvs(event, canvas)
        X.append(event.x)
        Y.append(event.y)
        print(event.x,event.y)
        
    canvas.bind("<ButtonPress-1>",printcoords) # Mouse clicking event. 
    plt.show()

    rt.mainloop()

def mylinfit(x,y):
    a = (sum(y-(statistics.mean(y))))/(sum(x-(statistics.mean(x))))   
    #a = (sum(y*x)-(sum(y)*sum(x))) / ((sum(x**2)-(sum(x))**2))
    b = ((statistics.mean(y))*sum(x) - (statistics.mean(x)*sum(y)))/(sum(x-statistics.mean(x)))
    #b = (sum(y))-(sum(y*x)*sum(x)-(sum(y)*sum(x)**2)) / (sum(x**2)-(sum(x))**2)
    return a,b
x = np.array(X)
y = np.array(Y)
a,b = mylinfit(x,y)
plt.plot(x,y,'kx')
xp = np.array(X)
plt.plot(xp,a*xp+b,'r-')
plt.show()
# -*- coding: utf-8 -*-


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as pp
import numpy as np
import random as rd

# constant
FILENAME = ["perceptron1", "perceptron2", "2Ccircle1", "2Circle1", "2Circle2", "2CloseS", "2CloseS2", "2CloseS3", "2cring", "2CS", "2Hcircle1", "2ring"]
COLOR = ['c', 'darkorange', 'm', 'lime', 'r']


# 輸出參考
def find_two_output(d):
    output = [d[0]]
    for i in d:
        if i != output[0]:
            output.append(i)
            break
    return output

# 畫圖
def draw(x, d, w_hide, w):
    # 畫點
    x1 = []
    x2 = []
    color = []
    for i in range(len(x)):
        x1.append(x[i][1])
        x2.append(x[i][2])
        color.append(COLOR[d[i]])
    x1_trans = x1
    x2_trans = x2
    for i in range(len(x)):
        x1_trans[i] = 1/(1+np.exp(-(-w_hide[0][0] + w_hide[0][1]*x1[i] + w_hide[0][2]*x2[i])))
        x2_trans[i] = 1/(1+np.exp(-(-w_hide[1][0] + w_hide[1][1]*x1[i] + w_hide[1][2]*x2[i])))
    pp.scatter(x1_trans, x2_trans, s=10, marker='o', color=color)
'''
    # 畫線
    line_x = np.linspace(min(x1_trans)-0.1, max(x1_trans)+0.1)
    line_y = (w[0]-w[1]*line_x)/w[2]
    pp.plot(line_x, line_y, 'k')
''' 
# 讀入資料
def read_file(file_name):
    f = open("DataSet/" + file_name + ".txt")
    x = []
    d = []
    for line in f:
        line = line.replace("\n", "")
        tmp = line.split(" ")
        x.append([float(tmp[0]), float(tmp[1])])
        d.append(int(tmp[2]))
    f.close()
    return [x, d]


def train(x, d, rate, recognition, times):
    # 初始化
    w_hide = [[-1.2, 1, 1], [0.3, 1, 1]]
    w = [0.5, 0.4, 0.8]
    y = [-1, 0, 0]
    delta = [0, 0, 0]
    OUTPUT = find_two_output(d)
    best_correct_rate = 0
    best_w = []
    best_w_hide = []
    
    # 調整輸入
    for i in range(len(x)):
        x[i] = [-1] + x[i]
    # 訓練
    for i in range(times):
        for j in range(len(x)):
            for k in range(1, 3):
                y[k] = 1/(1+np.exp(-(w_hide[k-1][0]*x[j][0] + w_hide[k-1][1]*x[j][1] + w_hide[k-1][2]*x[j][2])))
            result = 1/(1+np.exp(-(w[0]*y[0] + w[1]*y[1] + w[2]*y[2])))
            # 活化
            if result < 0.5:
                output = OUTPUT[0]
                # 倒傳遞
                if output != d[j]:
                    delta[2] = (1-result)*result*(1-result)
                    for k in range(1, 3):
                        delta[k-1] = y[k]*(1-y[k])*delta[2]*w[k]
                        for k in range(3):
                            for m in range(2):
                                w_hide[m][k] += rate*delta[m]*x[j][k]
                                w[k] += rate*delta[2]*y[k]
            else:
                output = OUTPUT[1]
                # 倒傳遞
                if output != d[j]:
                    delta[2] = (0-result)*result*(1-result)
                    for k in range(1, 3):
                        delta[k-1] = y[k]*(1-y[k])*delta[2]*w[k]
                        for k in range(3):
                            for m in range(2):
                                w_hide[m][k] += rate*delta[m]*x[j][k]
                                w[k] += rate*delta[2]*y[k]
        
        # 算正確率
        train_correct = 0
        for j in range(len(x)):
            for k in range(1, 3):
                y[k] = 1/(1+np.exp(-(w_hide[k-1][0]*x[j][0] + w_hide[k-1][1]*x[j][1] + w_hide[k-1][2]*x[j][2])))
            result = 1/(1+np.exp(-(w[0]*y[0] + w[1]*y[1] + w[2]*y[2])))
            # 活化
            if result < 0.5:
                output = OUTPUT[0]
            else:
                output = OUTPUT[1]
            if output == d[j]:
                train_correct += 1
        correct_rate = train_correct/len(x)
        if correct_rate > best_correct_rate:
            best_correct_rate = correct_rate
            best_w_hide = w_hide
            best_w = w
        print("echo {} correct_rate = {}".format(i+1, correct_rate))
        print("    w1 = {}".format(w_hide[0]))
        print("    w2 = {}".format(w_hide[1]))
        print("    w3 = {}".format(w))
        if correct_rate >= recognition:
            break
    return [best_w_hide, best_w, best_correct_rate, OUTPUT]

def test(x, d, w_hide, w, OUTPUT):
    # 初始化
    test_correct = 0
    y = [-1, 0, 0]
    RMSE = 0
    # 調整輸入
    for i in range(len(x)):
        x[i] = [-1] + x[i]
    # 測試正確率
    for i in range(len(x)):
        for k in range(1, 3):
            y[k] = 1/(1+np.exp(-(w_hide[k-1][0]*x[i][0] + w_hide[k-1][1]*x[i][1] + w_hide[k-1][2]*x[i][2])))
        result = 1/(1+np.exp(-(w[0]*y[0] + w[1]*y[1] + w[2]*y[2])))
        # 算均方差
        if d[i] == OUTPUT[0]:
            RMSE += result**2
        else:
            RMSE += (result-1)**2
        if result < 0.5:
            output = OUTPUT[0]
        else:
            output = OUTPUT[1]
        if output == d[i]:
            test_correct += 1
    RMSE = (RMSE/len(x))**0.5
    return [test_correct/len(x), RMSE]

# 按下輸入按鈕
def start():
    # 關閉前次訓練結果
    global pic_num
    if pic_num > 0: pp.close()
    pic_num += 1
    
    # 輸入
    rate = float(rate_entry.get())
    recognition = float(recognition_entry.get())
    times = int(times_entry.get())
    file_name = data_combo.get()
    
    # 資料分 2:1 做 訓練 & 測試
    [x, d] = read_file(file_name)
    n = int(len(x)/3)
    test_x = []
    test_d = []
    for i in range(n):
        rnd = rd.randint(0, len(x)-1)
        test_x.append(x.pop(rnd))
        test_d.append(d.pop(rnd))

    # 訓練
    [w_hide, w, train_correct_rate, OUTPUT] = train(x, d, rate, recognition, times)
    pp.subplot(211)
    draw(x, d, w_hide, w)

    # 測試
    [test_correct_rate, RMSE] = test(test_x, test_d, w_hide, w, OUTPUT)
    pp.subplot(212)
    draw(test_x, test_d, w_hide, w)

    # 輸出（訓練辨識率、測試辨識率、鍵結值
    for i in range(3):
        w[i] = round(w[i], 3)
        w_hide[0][i] = round(w_hide[0][i], 3)
        w_hide[1][i] = round(w_hide[1][i], 3)
    result = '{}\n訓練辨識率 = {:.3f}\n測試辨識率 = {:.3f}\n鍵結值：\n{}\n{}\n{}\nRMSE = {}'.format(file_name, train_correct_rate, test_correct_rate, w_hide[0], w_hide[1], w, RMSE)
    result_label.configure(text=result)
    pp.show()


# 開新視窗
window = tk.Tk()
# 設計視窗
window.title('107502508')
window.geometry('300x270')
window.configure(background='white')
# 標題
header_label = tk.Label(window, text='感知機類神經網路')
header_label.pack()

# 學習率 rate 群組
rate_frame = tk.Frame(window)
rate_frame.pack(side=tk.TOP)
rate_label = tk.Label(rate_frame, text='學習率：')
rate_label.pack(side=tk.LEFT)
rate_entry = tk.Entry(rate_frame)
rate_entry.pack(side=tk.LEFT)
# 辨識率 recognition 群組
recognition_frame = tk.Frame(window)
recognition_frame.pack(side=tk.TOP)
recognition_label = tk.Label(recognition_frame, text='收斂辨識率：')
recognition_label.pack(side=tk.LEFT)
recognition_entry = tk.Entry(recognition_frame)
recognition_entry.pack(side=tk.LEFT)
# 疊代上限 times 群組
times_frame = tk.Frame(window)
times_frame.pack(side=tk.TOP)
times_label = tk.Label(times_frame, text='疊代上限：')
times_label.pack(side=tk.LEFT)
times_entry = tk.Entry(times_frame)
times_entry.pack(side=tk.LEFT)
# 輸入資料 data 群組
data_frame = tk.Frame(window)
data_frame.pack(side=tk.TOP)
data_label = tk.Label(data_frame, text='輸入檔案：')
data_label.pack(side=tk.LEFT)
data_combo = ttk.Combobox(data_frame, values=FILENAME, state="readonly")
data_combo.current(0)
data_combo.pack(side=tk.LEFT)

# 輸入按鈕
input_btn = tk.Button(window, text='輸入', command=start)
input_btn.pack()

# 輸出
result_label = tk.Label(window)
result_label.pack()

# 畫圖相關
pp.ion()
pic_num = 0

# 執行視窗
window.mainloop()




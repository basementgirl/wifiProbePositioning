import os
import pandas as pd
from sympy.geometry import Circle
from sympy import *
import numpy as np


#对于被多个探针感应到的手机，取信号最强的前三个
def top(self, n=3):
    self.sort("RSSI",ascending=False)
    self.index=range(0,len(self))
    return self[:n]

def get_location_internal(log_file):
    lat = 0.0
    lng = 0.0
    variance = 0.0
    filename = log_file
    columns = ["Timestamp","ProbeMac","SourceMac","DestinationMac","BSSID","FrameType","RSSI","Channel","SSID"]
    data=pd.read_csv('/home/hadoop/sdl/hdfs_data//'+filename,header=None, names=columns)

    data=data[["ProbeMac","SourceMac","RSSI"]]
    data["location_x"] = float(filename.split("-")[0])
    data["location_y"] = float(filename.split("-")[1].split("_")[0])
    data=data[data["ProbeMac"].isin(['e4956e410ac2','e4956e4e540a','e4956e410abd','e4956e410b4c','e4956e410b32','e4956e410ac0','e4956e4e53e4','e4956e410acf','e4956e4e53e7'])]

    probedf = pd.DataFrame({"ProbeMac":["e4956e410ac2","e4956e4e540a","e4956e410abd","e4956e410b4c","e4956e410b32","e4956e410ac0","e4956e4e53e4","e4956e410acf","e4956e4e53e7"],
                            "probeloc_x":[8.5,12.0,17.5,20.5,23.5,26.8,26.8,35.5,38.2],
                            "probeloc_y":[5.7,9.0,5.5,7.2,5.2,8.0,5.5,5.5,7.4]})

    data = pd.merge(data, probedf, how="left", on="ProbeMac")

    #求每个探针对每个手机的感应均值
    data= data.groupby(["SourceMac","ProbeMac"],as_index=False).mean()
    data['predict_dis_liner']=10**(-0.05*data['RSSI']-2)
    data= data.groupby(["SourceMac"],as_index = False).apply(top)

    #开始利用三角定位求坐标
    lst_x=[]
    lst_y=[]
    for i in range(5): # 表示第i个分组
        #对于每一个手机分三种情况讨论。被一个探针感应到、被两个探针感应到、被三个探针感应到时。
        #如果只被一个探针感应到，取探针坐标作。
        if len(data.ix[i])==1:
            lat=float(data.ix[i,0]['probeloc_x'])
            lng=float(data.ix[i,0]['probeloc_y'])

        if len(data.ix[i])==2:
            circle1=Circle(Point(data.ix[i,0]['probeloc_x'],data.ix[i,0]['probeloc_y']),data.ix[i,0]['predict_dis_liner'])
            circle2=Circle(Point(data.ix[i,1]['probeloc_x'],data.ix[i,1]['probeloc_y']),data.ix[i,1]['predict_dis_liner'])
            loc1 = circle1.intersection(circle2)
            if len(loc1)==2:
                lat=float((loc1[0][0]+loc1[1][0])/2)
                lng=float((loc1[0][1]+loc1[1][1])/2)
            elif len(loc1)==1:
                lat=float(loc1[0][0])
                lng=float(loc1[0][1])
            else:
                lat=float((data.ix[i,0]['probeloc_x']+data.ix[i,1]['probeloc_x'])/2)
                lat=float((data.ix[i,0]['probeloc_y']+data.ix[i,1]['probeloc_y'])/2)
        #如果被三个探针感应到，利用三点定位，求坐标。
        if len(data.ix[i])==3:
            circle1=Circle(Point(data.ix[i,0]['probeloc_x'],data.ix[i,0]['probeloc_y']),data.ix[i,0]['predict_dis_liner'])
            circle2=Circle(Point(data.ix[i,1]['probeloc_x'],data.ix[i,1]['probeloc_y']),data.ix[i,1]['predict_dis_liner'])
            circle3=Circle(Point(data.ix[i,2]['probeloc_x'],data.ix[i,2]['probeloc_y']),data.ix[i,2]['predict_dis_liner'])
            loc1 = circle1.intersection(circle2)
            if len(loc1)==2:
                d_c3_1 = ((loc1[0][0]-data.ix[i,2]['probeloc_x'])**2+(loc1[0][1]-data.ix[i,2]['probeloc_y'])**2)**0.5
                d_c3_2 = ((loc1[1][0]-data.ix[i,2]['probeloc_x'])**2+(loc1[1][1]-data.ix[i,2]['probeloc_y'])**2)**0.5
                if d_c3_1>=d_c3_2:
                    loc1_x = loc1[1][0]
                    loc1_y = loc1[1][1]
                elif d_c3_1<d_c3_2:
                    loc1_x = loc1[0][0]
                    loc1_y = loc1[0][1]
            loc2 = circle1.intersection(circle3)
            if len(loc2)==2:
                d_c2_1 = ((loc2[0][0]-data.ix[i,1]['probeloc_x'])**2+(loc2[0][1]-data.ix[i,1]['probeloc_y'])**2)**0.5
                d_c2_2 = ((loc2[1][0]-data.ix[i,1]['probeloc_x'])**2+(loc2[1][1]-data.ix[i,1]['probeloc_y'])**2)**0.5
                if d_c2_1>=d_c2_2:
                    loc2_x = loc2[1][0]
                    loc2_y = loc2[1][1]
                elif d_c2_1<d_c2_2:
                    loc2_x = loc2[0][0]
                    loc2_y = loc2[0][1]
            loc3 = circle2.intersection(circle3)
            if len(loc3)==2:
                d_c1_1 = ((loc3[0][0]-data.ix[i,0]['probeloc_x'])**2+(loc3[0][1]-data.ix[i,0]['probeloc_y'])**2)**0.5
                d_c1_2 = ((loc3[1][0]-data.ix[i,0]['probeloc_x'])**2+(loc3[1][1]-data.ix[i,0]['probeloc_y'])**2)**0.5
                if d_c1_1>=d_c1_2:
                    loc3_x = loc3[1][0]
                    loc3_y = loc3[1][1]
                elif d_c1_1<d_c1_2:
                    loc3_x = loc3[0][0]
                    loc3_y = loc3[0][1]
            if len(loc1)==2 and len(loc2)==2 and len(loc3)==2:
                lat = float((loc1_x+loc2_x+loc3_x)/3)
                lng = float((loc1_y+loc2_y+loc3_y)/3)
            else:
                lat = float((data.ix[i,0]['probeloc_x'] + data.ix[i,1]['probeloc_x']+ data.ix[i,2]['probeloc_x'])/3)
                lng = float((data.ix[i,0]['probeloc_y'] + data.ix[i,1]['probeloc_y'] + data.ix[i,2]['probeloc_y'])/3)
        lst_x.append(lat)
        lst_y.append(lng)
        lat = round(sum(lst_x)/len(lst_x),1)
        lng = round(sum(lst_y)/len(lst_y),1)
        variance = ((lat-data["location_x"][0][0])**2+(lng-data["location_y"][0][0])**2)**0.5
    print(lat,lng,variance)
    # Please implement location estimation here.

def get_location(log_file):
    # The log_file is just like the log files used for training, except it contains signals captured from another place.
    os.chdir("/home/hadoop/sdl/hdfs_data//")
    if not os.path.exists(log_file):
        print("Cannot find file: ".format(log_file))
    get_location_internal(log_file)


if __name__ == "__main__":
    get_location("33-5.5_941")
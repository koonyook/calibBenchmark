
import numpy as np

# load in qualisys points 
##############################
### Read Qualisys tsv file ###
##############################
import datetime

class MarkerRecord:

    def __init__(self,filepath):
        self.markerNameList = []
        self.record = []
        self.frameNo = []
        self.timestamp = []	 #in second 
        self.event = []	#tuple of (name,frame,time)
        counter=0

        f=open(filepath,'r')    #this file pointer will be closed after the read is done
        while(True):
            line=f.readline()
            if(line==''):
                f.close()
                break
            else:
                s=line.strip().split('\t')
                if(s[0]=="NO_OF_FRAMES"):
                    pass
                elif(s[0]=="NO_OF_CAMERAS"):
                    pass
                elif(s[0]=="NO_OF_MARKERS"):
                    pass
                elif(s[0]=="FREQUENCY"):
                    self.frequency=float(s[1])  #important
                elif(s[0]=="NO_OF_ANALOG"):
                    pass
                elif(s[0]=="ANALOG_FREQUENCY"):
                    pass
                elif(s[0]=="DESCRIPTION"):
                    self.description=s[1]
                elif(s[0]=="TIME_STAMP"):		#TESTED
                    #TIME_STAMP	2021-01-08, 20:34:56.157	111711.00139370
                    #the first section is starting time
                    #the second section is number of seconds since the computer was started.
                    date,time=s[1].split(',')
                    time=time.strip()
                    year,month,day=date.split('-')
                    hour,minute,second=time.split('.')[0].split(':')
                    self.startTime=datetime.datetime(int(year),int(month),int(day),int(hour),int(minute),int(second),0)
                elif(s[0]=="DATA_INCLUDED"):
                    pass
                elif(s[0]=="EVENT"):
                    self.event.append((s[1],int(s[2]),float(s[3])))
                elif(s[0]=="MARKER_NAMES"):	#read all marker's name
                    self.markerNameList=s[1:]
                    #log.debug(self.markerNameList)
                elif(s[0]=="Frame" and s[1]=="Time"):		#header above each column
                    pass
                elif(s[0]=="TRAJECTORY_TYPES"):         #new version of QTM export tsv with this non-meaningful line
                    pass
                else:
                    if(len(s)>len(self.markerNameList)*3):	#frame number and time in the first 2 columns
                        self.frameNo.append(int(s[0]))
                        self.timestamp.append(float(s[1]))
                        numbers=s[2:]
                    else:	#no frame number and timestamp info
                        numbers=s
                    
                    row={}
                    for i,markerName in enumerate(self.markerNameList):
                            if(numbers[i*3+0]=="NULL" and numbers[i*3+1]=="NULL" and numbers[i*3+2]=="NULL"):
                                continue	#just skip this marker (normally happen at the early or late frame)
                            else:
                                row[markerName]=np.array([
                                    float(numbers[i*3+0])/1000,
                                    float(numbers[i*3+1])/1000,
                                    float(numbers[i*3+2])/1000
                                ])
                    self.record.append(row)

        self.firstFrameNo=self.frameNo[0]   #this is not index, don't be confused
        self.firstFrameIndex=self.firstFrameNo-1
        if self.firstFrameNo>1:   #this record is cropped
            fillerSize=self.firstFrameNo-1
            self.record = [{}]*fillerSize + self.record
            self.frameNo = [None]*fillerSize + self.frameNo
            self.timestamp = [None]*fillerSize + self.timestamp

    def getFirstMarkerTrajectory(self): #convert record to a list of (3,) or None
        ans=[]
        m=self.markerNameList[0]
        for rowDict in self.record:
            if m in rowDict:
                ans.append(rowDict[m])
            else:
                ans.append(None)
        return ans
    
    def getMarkerTrajectory(self,markerName):
        ans=[]
        for rowDict in self.record:
            if markerName in rowDict:
                ans.append(rowDict[markerName])
            else:
                ans.append(None)
        return ans

    def getMarkerTrajectoryAsNumpyArray(self, markerName):
        ans=np.ones([len(self.record),3],dtype=np.float32)*-1 #(-1,-1,-1) by default, because this position cannot happen naturally
        valid=np.zeros([len(self.record)],dtype=np.bool)
        for i,rowDict in enumerate(self.record):
            if markerName in rowDict:
                ans[i,:]=rowDict[markerName]
                valid[i]=True
        return ans,valid
    
    def getAllMarkerTrajectoryAsNumpyArray(self):
        ans={}
        valid={}
        for markerName in self.markerNameList:
            ans[markerName],valid[markerName]=self.getMarkerTrajectoryAsNumpyArray(markerName)
        return ans,valid

    def getAllMarkerTrajectoryAsOneNumpyArray(self,selectedMarkerList):
        markerN=len(selectedMarkerList)
        frameN=len(self.record)
        ans=np.ones((frameN,markerN,3),dtype=np.float32)*-1 #(-1,-1,-1) by default, because this position cannot happen naturally
        valid=np.zeros((frameN,markerN),dtype=np.bool)
        for i,rowDict in enumerate(self.record):
            for j,markerName in enumerate(selectedMarkerList):
                if markerName in rowDict:
                    ans[i,j,:]=rowDict[markerName]
                    valid[i,j]=True
        return ans,valid


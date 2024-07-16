import glob
import pickle
from typing import List

import numpy as np
import cv2

from tsvReader import MarkerRecord
import rigidAlignment

exposureTable={
     0 :0.786415,
    -1 :0.499823,
    -2 :0.249907,
    -3 :0.124951,
    -4 :0.062470,
    -5 :0.031186,
    -6 :0.015588,
    -7 :0.007787,
    -8 :0.003888,   #this is smaller than 2^-8
    -9 :0.001992,
    -10:0.000996,   #this is larger than 2^-10
    -11:0.000492,
    -12:0.000192,
    -13:0.000096,
    -14:0.000048,
}

triggerToStartExposureTime=0.00006969

def getHalfExposure(exposure):  
    #exposure can be from 0 to -13
    #the formula is unknown, but I can get it from excel sheet
    return exposureTable[exposure]/2

def extractQualisysAtGcamMidExposure(pos,valid,qualisysFrequency,gcamPeriod,triggerToStartExposureTime,halfExposure): 
    qualisysPeriod=1/qualisysFrequency

    qn=pos.shape[0]
    innerSize=len(pos.shape)-2

    fn=int(qn*qualisysPeriod//gcamPeriod)   #number of gcamFrame

    raisingEdgeIndex=np.arange(fn)
    t=raisingEdgeIndex*gcamPeriod + triggerToStartExposureTime + halfExposure   #this is the key different
    qLeftIndex=(t//qualisysPeriod).astype(int)
    qRightIndex = qLeftIndex+1

    leftPos=pos[qLeftIndex]
    rightPos=pos[qRightIndex]

    leftValid=valid[qLeftIndex]
    rightValid=valid[qRightIndex]
    newValid=np.logical_and(leftValid,rightValid)

    qRealIndex = t/qualisysPeriod
    rightRatio = qRealIndex-qLeftIndex
    leftRatio = 1-rightRatio

    expandAxisParam=[1]
    for i in range(innerSize):
        expandAxisParam.append(i+2)

    newPos = leftPos*np.expand_dims(leftRatio,axis=expandAxisParam) + rightPos*np.expand_dims(rightRatio,axis=expandAxisParam)    
    #neutralize the non-valid interpolation (make it zero)
    newPos = newPos * np.expand_dims(newValid, axis=expandAxisParam)   #len(newValid.shape)) 

    return newPos,newValid

def homoTranInv(mat):
    ans = np.eye(4)
    RT=mat[:3,:3].T
    ans[:3,:3]=RT
    ans[:3,3]=-RT @ mat[:3,3]
    return ans 

stopCriteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1e-4)

def RDB_triangulation(uv:np.ndarray, valid:np.ndarray, calibList:List):
    'Adapted from Triangulation2'
    valid_byframe = np.sum(valid,axis=1)
    validTriangulation:np.ndarray=(valid_byframe>=2)
    available_points = np.where(validTriangulation)[0]
    rejected_points = np.where(valid_byframe<2)[0]
    valid[rejected_points] = False

    if len(available_points)==0:
        triangulatedPoints=np.zeros((validTriangulation.shape[0],3),dtype=np.float32)
        AvgRaySqDist = np.zeros(valid.shape[0])-1   #-1 is impossible value
        AvgRayDist=-1   #-1 is impossible value
        rms=-1  #-1 is impossible value
        return triangulatedPoints,validTriangulation, AvgRaySqDist, AvgRayDist, rms, valid

    cn = len(calibList)
    total_frames = uv.shape[0]
    QList = np.zeros((cn,total_frames,3,3))
    cTQcList = np.zeros((cn,total_frames))
    QcList = np.zeros((cn,total_frames,3))

    for ci, calib in enumerate(calibList):
        valid_ci = np.where(valid[:,ci])[0]
        #Ensure that there is at least one point to triangulate
        if len(valid_ci)>0:
            qualisysFrame_camFrame = homoTranInv(calib['camFrame_labFrame'])
            #Undistort the points, Size will be n,2 --> 2,n  #these two values are normalized (Z=1)
            x_undistorted, y_undistorted = cv2.undistortPointsIter(uv[:,ci:ci+1,:], calib['K'], calib['D'], None, None, stopCriteria)[:,0].T
            #Will be of shape (3, total_frames)
            camFrame_ray = np.concatenate((x_undistorted.reshape(total_frames,-1),y_undistorted.reshape(total_frames,-1),[[1]]*total_frames),axis=1).T
            #(3,3) @ (3, total_frame) = (3, total_frame)
            qualisysFrame_Ray = qualisysFrame_camFrame[:3,:3] @ camFrame_ray
            c = qualisysFrame_camFrame[:3,3]
            #(total_frame, 3, 1)
            d = qualisysFrame_Ray/np.linalg.norm(qualisysFrame_Ray,axis=0)
            identity = np.array([np.eye(3)]*total_frames)
            #(total_frame,3) @ (total_frame,3) -> (total_frame,3,3)
            Q = identity - np.einsum('ji,ki->ijk',d,d)
            QList[ci,valid_ci] = Q[valid_ci]
            #(total_frame,3,3) @ (total_frame,3) -> (total_frame,3)
            Qc = Q@c
            QcList[ci,valid_ci] = Qc[valid_ci]
            #(3,) @ (3,total_frame) -> (total_frame)
            cTQcList[ci,valid_ci] = (c@Qc.T)[valid_ci]

    sumQ = np.sum(QList,axis=0)
    sumQc = np.sum(QcList,axis=0)
    sum_cTQc = np.sum(cTQcList,axis=0)

    #to avoid singularity error during the solve, replace any sumQ that has determinant of 0 with simple identity matrix
    det=np.linalg.det(sumQ)
    zeroDet=(det==0).reshape((total_frames,1,1))
    identityGrid=np.zeros((total_frames,3,3))
    identityGrid[:,0,0]=identityGrid[:,1,1]=identityGrid[:,2,2]=1
    sumQ=np.logical_not(zeroDet)*sumQ + zeroDet*identityGrid    #new sumQ should pass singular matrix problem
    
    triangulatedPoints:np.ndarray = np.linalg.solve(sumQ,sumQc)    #this is fast enough to do them all regardless of availability  (dv_count,3)
    
    SqDist = sum_cTQc[available_points] - np.einsum('ij,ij->i',sumQc[available_points],triangulatedPoints[available_points])
    
    AvgRaySqDist = np.zeros(valid.shape[0])
    AvgRaySqDist[available_points] = SqDist/valid_byframe[available_points]

    #Calculate the average overall distance
    AvgRayDist = np.sqrt(np.sum(AvgRaySqDist)/len(available_points))
    #Calculate the RMS
    AvgRaySqDist[(AvgRaySqDist<0) & (AvgRaySqDist>-1e-8)] = 0
    rms = np.sqrt(AvgRaySqDist)
    triangulatedPoints=triangulatedPoints.astype(np.float32)
    return triangulatedPoints,validTriangulation, AvgRaySqDist, AvgRayDist, rms, valid

def getExtrinsicMatrixFromRvecTvec(rvec,tvec):
    camFrame_objectFrame=np.eye(4)
    camFrame_objectFrame[:3,:3]=cv2.Rodrigues(rvec)[0]
    camFrame_objectFrame[:3,3:4]=tvec.reshape(3,1)   #(3,1)
    return camFrame_objectFrame

def modCalibFormat(oneCamDict):
    oneCamDict['D']=oneCamDict['D'].reshape((1,5))
    oneCamDict['camFrame_labFrame']=getExtrinsicMatrixFromRvecTvec(oneCamDict['rvec'], oneCamDict['tvec'])
    return oneCamDict

class Evaluator:

    def __init__(self,coreFolder):
        with open(coreFolder+'metadata.pkl','rb') as f:
            self.metaDict=pickle.load(f)
        
        #load qualisys point
        self.halfExposure=getHalfExposure(self.metaDict['exposure'])
        tmp=glob.glob(coreFolder+"*.tsv")
        if len(tmp)==1:
            mr=MarkerRecord(tmp[0])
        else:
            exit()

        self.qualisysData=mr
        self.qualisysFrequency=mr.frequency

        #load uv files
        camIdList=self.metaDict['camIdList']

        with open(coreFolder+'centroidsUV%s.pkl'%camIdList[0],'rb') as f:
            frameN=len(pickle.load(f))

        camN=len(camIdList)
        uv=np.zeros([frameN,camN,2])
        valid=np.zeros([frameN,camN],dtype=bool)
        for ci,camId in enumerate(camIdList):
            with open(coreFolder+'centroidsUV%s.pkl'%camId,'rb') as f:
                tmp=pickle.load(f)
            for fi in range(frameN):
                content=tmp[fi]
                if content is not None:
                    uv[fi,ci,:]=content
                    valid[fi,ci]=True
        
        self.camIdList=camIdList
        self.uv=uv
        self.valid=valid
        self.frameN=frameN

        qualisysFrequencyDivisor=self.metaDict['qualisysFrequencyDivisor']
        camPeriod=qualisysFrequencyDivisor/self.qualisysFrequency

        qualisysPoints,qualisysValid=self.qualisysData.getMarkerTrajectoryAsNumpyArray('marker')
        self.interpolatedQualisysPoints,self.interpolatedQualisysValid=extractQualisysAtGcamMidExposure(qualisysPoints,qualisysValid,self.qualisysFrequency,camPeriod,triggerToStartExposureTime,self.halfExposure)


    def triangulateAndGetAvgError(self,calibDict):
        calibList=[]
        for camId in self.camIdList:
            calibList.append(modCalibFormat(calibDict[camId]))

        triangulatedPoints,validTriangulation, AvgRaySqDist, AvgRayDist, rms, validOut = RDB_triangulation(self.uv,np.copy(self.valid),calibList)

        assert(np.sum(validTriangulation)==self.frameN)

        errAvg,errMax,c=rigidAlignment.alignAndGetError(triangulatedPoints,self.interpolatedQualisysPoints,allowScaling=False)

        return errAvg
        


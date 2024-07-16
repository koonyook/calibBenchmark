import numpy as np

def rigid_transform_3D(A, B, allowScaling=False):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    n=num_cols

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    varA = np.var(Am, axis=1)
    c = S.sum()/varA.sum()/n # scale factor

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    if allowScaling:
        t = -c*R @ centroid_A + centroid_B
        return R, t, c
    else:
        t = -R @ centroid_A + centroid_B
        return R, t, 1

def alignAndGetError(a,b,allowScaling=False):
    minN=min(a.shape[0],b.shape[0])
    a=a[:minN,:]
    b=b[:minN,:]
    R,t,c=rigid_transform_3D(a.T,b.T,allowScaling)  
    
    transformedA=(c*R@a.T+t).T
    distances=np.linalg.norm(transformedA-b,axis=1) 
    avgError=np.average(distances)
    maxError=np.max(distances)
    return avgError,maxError,c

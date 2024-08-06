
# Helper files
# Authors: Nezih Topaloglu and Ahmet Agaoglu

import numpy as np
import cv2





def calc_real_mae(yi, yf, yi_gt, yf_gt, width):
    '''
    Calculate the mean absolute error (MAE) between the ground truth and the estimated horizon line.
    Inputs:
    - yi: the initial vertical coordinate of the estimated horizon line
    - yf: the final vertical coordinate of the estimated horizon line
    - yi_gt: the initial vertical coordinate of the ground truth horizon line
    - yf_gt: the final vertical coordinate of the ground truth horizon line
    - width: the width of the image

    Output:
    - mae: the mean absolute error between the ground truth and the estimated horizon
    '''

    begin_error = yi - yi_gt
    end_error = yf - yf_gt

    if begin_error == end_error:
        return np.abs(begin_error)
    error = np.arange(begin_error, end_error, (end_error - begin_error) / width)
    abs_error = np.abs(error)
    mae = np.mean(abs_error)

    return mae



def convert_yi_yf_to_y_theta(yi, yf, w, degree = False):
    '''
    Convert the coordinates to the form (y, theta) where y is the vertical coordinate and theta is the angle in degrees
    Inputs:
    - yi: the initial vertical coordinate of the horizon line
    - yf: the final vertical coordinate of the horizon line
    - w: the width of the image
    - degree: if True, the angle is returned in degrees, otherwise in radians

    Output:
    - y: the vertical coordinate of the horizon line midpoint
    - theta: the angle of the horizon line in degrees or radians
    '''

    # 
    y = (yi + yf) / 2
    theta = np.arctan((yf - yi) / w)
    if degree:
        theta = np.degrees(theta)
    return int(y), theta


def convert_y_theta_to_yi_yf(y, theta, w, degree = False):
    '''
    Convert the coordinates to the form (yi, yf) where yi and yf are the vertical coordinates
    This function is not used in the project, but it is provided for completeness
    Inputs:
    - y: the vertical coordinate of the horizon line midpoint
    - theta: the angle of the horizon line in degrees or radians
    - w: the width of the image
    - degree: if True, the angle is in degrees, otherwise in radians

    Output:
    - yi: the initial vertical coordinate of the horizon line
    - yf: the final vertical coordinate of the horizon line
    '''
    
    if degree:
        theta = np.radians(theta)
    # Convert the coordinates to the form (yi, yf) where yi and yf are the vertical coordinates:
    yi = y - w * np.tan(theta)/2
    yf = y + w * np.tan(theta)/2
    return int(yi), int(yf)


    
def bgr2hsv(rgb):
    '''
    Convert the BGR image to the HSV color space
    Inputs:
    - rgb: the BGR image

    Output:
    - hsv: the HSV image
    '''
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)



def error_estimate(x, frame_width, frame_height, alpha, I1, I2, I3, Xmat, Ymat):
    '''
    This function estimates the error metric. It uses the variance of the pixels in the sky and sea regions.
    Inputs:
    - x: the parameters of the horizon line (y, theta)
    - frame_width: the width of the image
    - frame_height: the height of the image
    - alpha: the ratio of the area of the paralloelogram (sea or sky) to the height of the image
    - I1, I2, I3: the three channels of the image
    - Xmat, Ymat: the meshgrid matrices for the image
    '''

    yI = x[0]
    theta = np.radians(x[1])
    
    tan_theta = np.tan(theta)

    # The leftmost pixel of the horizon line:
    c = yI - tan_theta * frame_width/2

    # The height of the parallelogram (A_sky or A_sea):
    delta_h = alpha * frame_height

    #The pixels constituting the horizon line:
    ind_ref = tan_theta * Xmat + c

    # The pixels constituting the upper and lower edges, in Fig.4 of the paper:
    ind_lower = ind_ref - delta_h
    ind_upper = ind_ref + delta_h

    # The pixels corresponding to the sky and sea regions:
    ind_sky = (Ymat < ind_ref) & (Ymat > ind_lower)
    ind_sea = (Ymat > ind_ref) & (Ymat < ind_upper)

    
    area_sky = np.sum(ind_sky)
    area_sea = np.sum(ind_sea)

    # The normed variance of the pixels in the sky and sea regions
    n_sky = np.linalg.norm([np.var(I1[ind_sky]), np.var(I2[ind_sky]), np.var(I3[ind_sky])])
    n_sea = np.linalg.norm([np.var(I1[ind_sea]), np.var(I2[ind_sea]), np.var(I3[ind_sea])])

    # Finally, the error estimate:
    f = n_sky * area_sky / (area_sky + area_sea) + n_sea * area_sea / (area_sky + area_sea)

    return f



# The Intensity-based Horizon Line Detection Algorithm (HLDA) is implemented in this file.
# The algorithm is based on the paper 
# "An intensity-difference-based maritime horizon detection algorithm" by Nezih Topaloglu, 2024.
# The algorithm is implemented in the detectHL function.

# Author: Nezih Topaloglu


import numpy as np
import cv2
import os
from sys import platform


from datetime import datetime
from helper import *



def line_search_and_fit(x_coords, max_grad_matrix, width):
    '''
    This function applies the "line searching and fitting" algorithm, as described in the paper.
    The function takes the x_coords, max_grad_matrix (L matrix after reduction), and the width of the image as input.
    '''


    def fit_and_calc(x_coords,y_final):
        '''
        Fit a line to the data and calculate the mean absolute error.
        '''

        # Fit a line to the filtered data, using least squares:
        fit = np.polyfit(x_coords, y_final, 1)

        m, b = fit
        residuals = y_final - (m*x_coords + b)

        # Calculate the mean absolute error:
        mean_abs_error = np.mean(np.abs(residuals))
               
        # Also calculate R^2, using slope and intercept from the fit:
        # Note that R^2 is not used.
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_final - np.mean(y_final))**2) + 1e-9
        r_squared = 1 - (ss_res / ss_tot)

        return r_squared, mean_abs_error, residuals, fit

    def calc_yi_yf_ymean(fit, width):
        '''
        Calculate the yi, yf, and ymean values from the fit.
        '''

        yi = int(np.polyval(fit, 0))
        yf = int(np.polyval(fit, width))

        return yi, yf, int((yi+yf)/2)

    def get_next_highest(max_grad_matrix,max_point_idx, indices):
        '''
        Get the next highest point in the corresponding column. In here, highest means the 
        highest pixel value in the height direction. Therefore, the next highest point is the
        next lowest locationed point.
        '''

        # Choose the next highest point in the corresponding column:
        idx = indices[max_point_idx]
        
        if idx == 0: # If the index is zero, it means the highest point is reached, so return False:
            
            return False, max_grad_matrix[max_point_idx,idx], indices

        indices[max_point_idx] -= 1
        idx = indices[max_point_idx]

        if np.isnan(max_grad_matrix[max_point_idx,idx]):

            res1, res2, res3 = get_next_highest(max_grad_matrix,max_point_idx, indices)
            return res1, res2, res3
        else:
            return True, max_grad_matrix[max_point_idx,idx], indices

    max_iter = 1000

    #Get the number of maxes:
    num_maxes = max_grad_matrix.shape[1]

    # Sort each column from smallest to largest:
    max_grad_matrix = np.sort(max_grad_matrix, axis=1)

    # Find the column index of the first occuring NaN in each row, in max_grad_matrix:
    # NaN values correspond to the limits.
    nan_indices = np.argmax(np.isnan(max_grad_matrix), axis=1)

    #Starting from the lowest point in each column:
    points = max_grad_matrix[np.arange(len(nan_indices)), nan_indices-1]



    indices = np.ones_like(points)*(num_maxes-1)
    indices = indices.astype(int)

  
    mae = 1e6 # Initialize the mean absolute error to a large value

    # Fit a line to the points:
    _, mae, _, fit = fit_and_calc(x_coords, points)

    #Update the best MAE, fit and points values:
    mae_hgh_best = mae
    fit_hgh_best = fit
    points_hgh_best = points.copy()

    iter = 0

    k = 1

    while mae > 2 and iter < max_iter:

        # Find the highest point in points:


        if k <= len(points):
            max_point_idx = np.argsort(points)[-k]
        else:
            break

        
        # Choose the next highest point in the corresponding column:
        res, next_point, indices = get_next_highest(max_grad_matrix,max_point_idx, indices)

        if res: # Next highest point is found, update the points array:
            points[max_point_idx] = next_point
        else: # If no next highest point is found, continue to the next point in the sorted points array:
            k += 1
            continue

        # Fit a line to the points:
        _, mae, _, fit = fit_and_calc(x_coords, points)

        # Update the best MAE, fit and points values if the new MAE is better:
        if mae < mae_hgh_best:
            mae_hgh_best = mae
            fit_hgh_best = fit.copy()
            points_hgh_best = points.copy()


        iter += 1


    # Choose the best fit, MAE, and points values:
    fit_best = fit_hgh_best
    mae_best = mae_hgh_best
    points = points_hgh_best

    # Calculate the yi, yf, and ymean values from the fit:
    yi,yf,ymean = calc_yi_yf_ymean(fit_best, width)

 
    return yi, yf, ymean, mae_best, points




def detectHL(image, params):

    '''
    This function applies the HLDA algorithm on the input image.
    The function takes the image and the parameters dictionary as input.
    The parameters dictionary should contain the following keys:
    - no_strips: Number of strips to divide the image horizontally
    - no_cells: Number of cells to divide the strips vertically
    - overlap_ratio: Overlap ratio between the cells
    - num_maxes: Number of maxes to consider in the L matrix
    '''

    # Receive the parameters from the params dictionary:
    no_strips = params.get('no_strips', 10)
    no_cells = params.get('no_cells', 10)
    overlap_ratio = params.get('overlap_ratio',0.5)
    num_maxes = params.get('num_maxes', 5)

    # Get the image dimensions:
    height, width, _ = image.shape

    # Calculate the strip width and cell height:
    strip_width = width // no_strips
    cell_height = height // no_cells

    # Calculate the x coordinates of the strips:
    x_coords = np.arange(no_strips) * strip_width


    image_orig = image.copy()

    # Apply RGB BGR correction on image_orig:
    image_orig[:,:,0] = image[:,:,2]
    image_orig[:,:,2] = image[:,:,0]


    # Apply the median filter 2D, with a Kernel size of 25x25 to the image:
    image = cv2.medianBlur(image, 25)

    # Calculate the step size for the overlapping cells
    step_size = int(cell_height * (1 - overlap_ratio))

    # Calculate the number of iterations:    
    num_iterations = (height - cell_height)//step_size

    # Create the gradients_matrix, denoted as D matrix in the paper:
    gradients_matrix = np.zeros((no_strips, num_iterations))
    
    # Create the max_grad_matrix, denoted as L matrix in the paper:
    max_grad_matrix = np.zeros((no_strips, num_maxes))

    ups = []
    downs = []


    # Loop through the strips:
    for strip in range(no_strips):
        strip_x1 = strip * strip_width
        strip_x2 = min((strip + 1) * strip_width, width)  # Handle the last strip if it's smaller

        strip_image = image[:, strip_x1:strip_x2]

        strip_y_coords = []

   

        for cell_y1 in range(0, height - cell_height + 1, step_size):
            cell_y2 = cell_y1 + cell_height

            cell_region = strip_image[cell_y1:cell_y2, :]

            # Calculate the mean color vector for the cell
            mean_color = np.mean(cell_region, axis=(0, 1))

            # Store the mean color vector
            strip_y_coords.append(mean_color)

        # Calculate the gradient of the mean color vectors
        strip_y_coords = np.array(strip_y_coords)
        gradients = np.linalg.norm(np.diff(strip_y_coords, axis=0), axis=1)

        gradients_matrix[strip,:] = gradients # Store the gradients in the gradients_matrix


        sorted_gradients = np.argsort(gradients) #Sort the gradients



        # Copy the max num_maxes values of gradients into max_grad_matrix:
        max_grad_matrix[strip,:] = sorted_gradients[-num_maxes:]

        
        # Sort from smallest to largest:
        max_grad_matrix[strip,:] = np.sort(max_grad_matrix[strip,:])

        # Initialize the consecutive_change_count, used in the Reduction step in the paper:
        consecutive_change_count = 0


        # If there are two consecutive values in max_grad_matrix[strip,:], like 15 and 16, replace the first one with 16, so they become 16 and 16:
        for i in range(max_grad_matrix.shape[1]-1):

            if max_grad_matrix[strip,i] == max_grad_matrix[strip,i+1]-1:
                max_grad_matrix[strip,i] = max_grad_matrix[strip,i+1]
                # Also go back, from i-1 to 0, and check the previous values:
                for j in range(i-1,-1,-1):
                    if max_grad_matrix[strip,j] == max_grad_matrix[strip,j+1]-1:
                        max_grad_matrix[strip,j] = max_grad_matrix[strip,j+1]
                        consecutive_change_count += 1
                    else:
                        break



        max_grad_matrix[strip,:] = ((max_grad_matrix[strip,:]+0.75)*step_size).astype(int)



        strip_values = max_grad_matrix[strip,:]

        strip_values_min = np.min(strip_values)
        strip_values_max = np.max(strip_values)
        ups.append(strip_values_min)
        downs.append(strip_values_max)


    ups_mean = np.min(ups)
    downs_mean = np.max(downs)

    # Prepare a new matrix for the filtered max_grad_matrix:
    filtered_max_grad_matrix = np.zeros(max_grad_matrix.shape)
    for i in range(max_grad_matrix.shape[0]):
        for j in range(max_grad_matrix.shape[1]):
            if max_grad_matrix[i,j] >= ups_mean and max_grad_matrix[i,j] <= downs_mean:
                filtered_max_grad_matrix[i,j] = max_grad_matrix[i,j]
            else:
                filtered_max_grad_matrix[i,j] = np.nan

    # If one row is all nan, remove it from the matrix and x_coords:
    x_coords = x_coords[~np.isnan(filtered_max_grad_matrix).all(axis=1)]
    filtered_max_grad_matrix = filtered_max_grad_matrix[~np.isnan(filtered_max_grad_matrix).all(axis=1)]    
    

    # Apply the line_search_and_fit function to get the yi and yf:
    yi, yf, _, _, points = line_search_and_fit(x_coords, filtered_max_grad_matrix, width)

    # The actual output is yi and yf, the rest is returned for debugging and visualisation purposes:
    return ups, downs, x_coords, max_grad_matrix, yi, yf, points



def test_detectHL(video_url, output_path):
    '''
    Test the detectHL function on a video file.
    The function takes the video URL and the output path as input.
    It processes the video frame by frame, and applies the detectHL function on each frame.
    The output is saved as a video file in the output path.
    '''

    # Open the video file:
    cap = cv2.VideoCapture(video_url)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output the video in XVID format:
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))


    frame_count = 0
    


    while cap.isOpened():
        ret, frame = cap.read()

        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
        frame_count += 1
        if not ret:
            break

        # Apply the detectHL function on the frame:
        ups, downs, x_coords, max_grad_matrix, yi_final, yf_final, points = detectHL(frame, 
                                                                params={'no_strips': 20, 
                                                                        'no_cells': 15, 
                                                                        'num_maxes': 6,
                                                                        'overlap_ratio':0.5,
                                                                        'height_coeff': 0.2,})

        # Draw a yellow line to represent the horizon
        ups = np.array(ups).astype(int)
        downs = np.array(downs).astype(int)
        x_coords = x_coords.astype(int)


        mean_ups = np.mean(ups)
        mean_downs = np.mean(downs)

        cv2.line(frame, (0, int(mean_ups)), (frame_width, int(mean_ups)), (0, 0, 255), 3)
        cv2.line(frame, (0, int(mean_downs)), (frame_width, int(mean_downs)), (0, 0, 255), 3)
        cv2.line(frame, (0, yi_final), (frame_width, yf_final), (0, 255, 255), 2)
        

        #Plot the x_coords and y_coords_all as small circles on the frame
        for i in range(x_coords.shape[0]):
            for j in range(max_grad_matrix.shape[1]):
                if not np.isnan(max_grad_matrix[i,j]):
                    cv2.circle(frame, (int(x_coords[i]), int(max_grad_matrix[i,j])), 6, (0, 0, 255), -1)

        # Plot the points as small circles on the frame
        for i in range(points.shape[0]):
            cv2.circle(frame, (int(x_coords[i]), int(points[i])), 6, (0, 0, 255), -1)

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and the output video
    cap.release()
    out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    # Test the function with a sample video URL:
    
    input_video_url = 'Buoy/Videos/buoyGT_2_5_3_0.avi'



    now = datetime.now()
    nowStr = now.strftime("%d%m%Y%H%M")

  

    if platform!= 'linux':
        input_video_url = 'Buoy\\Videos\\buoyGT_2_5_3_0.avi'

    else:
        input_video_url = 'Buoy/Videos/buoyGT_2_5_3_0.avi'
    
    
    outputURL = 'output_HLDA_' + nowStr + '_.mp4'

    test_detectHL(input_video_url, outputURL)

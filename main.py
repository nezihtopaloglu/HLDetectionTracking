
# The main file to run the On-Off Horizon Line Detection algorithm with the Intensity-Difference-based HL Detection Algorithm
# Two datasets are used: Singapore-onboard and Buoy

# Authors: Nezih Topaloglu and Ahmet Agaoglu

#Importing libraries
import numpy as np
import cv2

import os
from sys import platform
from os import listdir
import matplotlib
from datetime import datetime
import time

# Importing the helper functions and the HLDA functions:
from helper import *
from HLDA import *


# Choose the dataset:

#dataset = "Singapore-onboard"
dataset = "Buoy"

# Choose whether to save the output video or not:
save_output_video = False


#----------------------------------------------
# On-Off control parameters:

# The threshold array used in the On-Off control, a single value can be used as well:
threshold_array = [0.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8]
forecast_steps = 20  # Number of steps to forecast
ratio = 0.1 # Ratio of the frame height
alpha = 0.1 # Alpha value for the error calculation
#----------------------------------------------

#----------------------------------------------
#Intensity-Difference-based HL Detection Algorithm parameters:
no_strips = 25
no_cells = 20
num_maxes = 8
overlap_ratio = 0.5
height_coeff = 0.2

#----------------------------------------------

# CSV filenames:
csv_filename_avg = "averages.csv" #The CSV file that will store the averages for the chosen dataset
csv_filename = "results.csv" #The CSV file that will store the results for each video

# Check if the file exists:
if not os.path.exists(csv_filename):
    # If it does not exist, create it and write the header:
    with open(csv_filename, 'w') as f:
        f.write("Dataset,filename,nowStr,ratio,threshold, save_vid, full_roi_count, mean_real_err,std_real_err,mean_roi_diff_training_removed_percent,std_roi_diff_training_removed,mean_error,std_error,elapsed, total_time, num_incl,num_excl, forecast_steps,total_frames,frame_width,frame_height,input_video_url\n")
    
if not os.path.exists(csv_filename_avg):    
    with open(csv_filename_avg, 'w') as f:
        f.write("Dataset,filename,nowStr,ratio,threshold, save_vid, full_roi_count, mean_real_err,std_real_err,mean_roi_diff_training_removed_percent,std_roi_diff_training_removed,elapsed, total_time, num_incl,num_excl, forecast_steps,frame_width,frame_height\n")
      
with open(csv_filename, 'a') as f:
    f.write(f"Analysis starts with HLDA params: no_strips={no_strips} / no_cells={no_cells} / num_maxes={num_maxes} / overlap_ratio={overlap_ratio} / height_coeff={height_coeff}\n")



# Loop through the threshold array:
for threshold in threshold_array:

    mean_real_err_array = []
    std_real_err_array = []
    mean_roi_diff_training_removed_percent_array = []
    std_roi_diff_training_removed_array = []
    elapsed_array = []
    total_time_array = []
    full_roi_count_array = []
    num_inclusions_array = []
    num_exclusions_array = []




    # Starting timer:
    now = datetime.now()
    nowStr = now.strftime("%d%m%Y%H%M")


    # Make sure there are no open videowriters or windows:
    cv2.destroyAllWindows()


    # Choose the folder where the videos are stored:

    # If the platform is Linux:
    if platform == "linux" or platform == "linux2":
        if dataset == "Singapore-onboard":
            folder_files = "VIS_Onboard/Videos/"
        elif dataset == "Buoy":
            folder_files = "Buoy/Videos"

    # If the platform is Windows:
    elif platform == "win32" or platform == "win64":
        if dataset == "Singapore-onboard":
            folder_files = "VIS_Onboard\Videos"
        elif dataset == "Buoy":
            folder_files = "Buoy\Videos"
            

    print(f"Number of files: {len(listdir(folder_files))}")

    # Loop through all the files in the folder:
    for file in listdir(folder_files):
        num_inclusions = 0
        num_exclusions = 0
        if file.endswith(".avi"):
            input_video_url = folder_files + "/" + file
            filename = os.path.basename(input_video_url)


            if platform != "win32":
                matplotlib.use('GTK3Agg')


            #strip the extension from the filename:
            filename = os.path.splitext(filename)[0]

            # Check if the output_videos folder exists, if not create it:
            if not os.path.exists('output_videos'):
                os.makedirs('output_videos')


            outputURL = "output_videos/" + filename + "_" + nowStr + "_onoff.mp4"

            print(filename)

            # The ground truth (GT) .npz file URL:            
            npzURL = os.path.normpath(os.path.join(folder_files, os.pardir)) + "/HorizonGTCorrected/" + filename + "_GT.npz" # Using the corrected ground truth by A.Agaoglu

            # Check if the .npz file exists:
            if os.path.exists(npzURL):
                # If it exists, load the data:
                data = np.load(npzURL)
                yi_array_gt = data['yi_array']
                yf_array_gt = data['yf_array']
            else:
                print("The .npz file does not exist. Exiting...")
                continue


            cap = cv2.VideoCapture(input_video_url)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frame_rate = int(cap.get(5))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Frame width:", frame_width, "Frame height:", frame_height, "Frame rate:", frame_rate, "Total frames:", total_frames)


            if save_output_video:
                out = cv2.VideoWriter(outputURL, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))


            # Initialize the arrays and variables:

            frame_count = 0
            full_roi_count = 0


            base_yi = []
            base_yf = []

            base_y = []
            base_theta = []

            roi_diff_array = []
            real_err_array = []

            y_final_array = []
            t_final_array = []

            error = []

            elapsed_time_array = []
            n_iter_array = []
            HLDA_time_array = []

            yi_prev = int(frame_height/2)
            yf_prev = int(frame_height/2)

            y_prev = int(frame_height/2)
            theta_prev = 0

            roi_min = 0
            roi_max = frame_height

            t = np.arange(0, frame_width)
            H_true = np.zeros((total_frames, 2))

            #The coordinates of the full frame pixels:
            Xmat_or, Ymat_or = np.meshgrid(np.arange(0, frame_width), np.arange(0,frame_height))


            while cap.isOpened():


                roi_diff = roi_max-roi_min # Updatind the ROI height, denoted as roi_diff

                if roi_diff == frame_height:
                    full_roi_count += 1 # Increment the full_roi_count if the ROI height is equal to the frame height

                #Setting fixed lower and upper bounds:
                theta_limit = np.degrees(np.arctan((roi_max-roi_min)/(frame_width)))
                LB = [0,-theta_limit]
                UB = [roi_max-roi_min, theta_limit]

                
                #Half of the frame width (a parameter for the error calculation):
                x_ref = frame_width/2


                # Reading the frame:
                ret, frame = cap.read()

                if frame_count % 100 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}, threshold: {threshold}")

                if not ret:
                    break


                # Check if frame_count is greater than the length of the ground truth data:
                if frame_count >= len(yi_array_gt)-1:
                    break

                # Start timer:
                start = time.time()
                

                #Crop the frame to ROI:
                frame_to_process = frame[roi_min:roi_max, :,:]

                # Convert the frame to HSV:
                HSV = bgr2hsv(frame)
                H = HSV[:, :, 0]
                S = HSV[:, :, 1]
                V = HSV[:, :, 2]

                # The HSV channels of the cropped frame:
                H_to_process = H[roi_min:roi_max, :]
                S_to_process = S[roi_min:roi_max, :]
                V_to_process = V[roi_min:roi_max, :]
                

                # The coordinates of the cropped frame pixels:
                Xmat,Ymat = np.meshgrid(np.arange(0, frame_width),np.arange(0,roi_diff))


                HLDA_time_start = time.time()

                # HL detection using intensity-difference-based HLDA:
                ups, downs, x_coords, max_grad_matrix, yi_final, yf_final, points = detectHL(frame_to_process, 
                                                                        params={'no_strips': no_strips, 
                                                                                'no_cells': no_cells, 
                                                                                'num_maxes': num_maxes,
                                                                                'overlap_ratio':overlap_ratio,
                                                                                'height_coeff': height_coeff,})


                
                HLDA_time_end = time.time()
                HLDA_time = HLDA_time_end - HLDA_time_start
                HLDA_time_array.append(HLDA_time)
                
                #Transform the results from the ROI coordinates to the full frame coordinates:
                yi_final = yi_final + roi_min
                yf_final = yf_final + roi_min

                #Convert the yi_final and yf_final to y_final and theta_final:
                y_final, theta_final = convert_yi_yf_to_y_theta(yi_final, yf_final, frame_width, degree = True)

                    
                #Updating the H_true matrix with the calculated y_final and theta_final values:
                H_true[frame_count, :] = [y_final, theta_final]

                # Calculating the pixels on the horizon line:
                y = np.round(np.tan(np.radians(H_true[frame_count,1]))*t+H_true[frame_count,0])

                if threshold != 0:

                # Updating the error array with error_estimate:
                    error.append(error_estimate(H_true[frame_count, :], 
                                                frame_width,
                                                frame_height, 
                                                alpha, 
                                                H, 
                                                S, 
                                                V, 
                                                Xmat_or, 
                                                Ymat_or))
                else:
                    error.append(0) # If threshold is 0, the error estimation is not performed.

                #Print the y and theta values and the GT values every 50 frames:
                if frame_count%50 == 0:
                    print("y and theta:", y_final, theta_final,"yi and yf:", yi_final, yf_final, "y_gt:", yi_array_gt[frame_count], yf_array_gt[frame_count], "error:", error[-1])

                # Append the y_final and theta_final values to the arrays:
                y_final_array.append(y_final)
                t_final_array.append(theta_final)



                #Calculating the mean absolute error, the error compared to the ground truth:
                if dataset == "Singapore-onboard" :
                    real_mae = calc_real_mae(yi_final, yf_final, yi_array_gt[frame_count], yf_array_gt[frame_count], frame_width)
                elif dataset == "Buoy":
                    real_mae = calc_real_mae(yi_final, yf_final, yi_array_gt[frame_count][0][0], yf_array_gt[frame_count][0][0], frame_width)
                # print("real_mae:", real_mae)

                # IF error is not 1d, make it 1d:
                if len(real_mae.shape) > 0:
                    real_mae = real_mae[0]

                # Append the real error to the real_err_array:
                real_err_array.append(real_mae)

                #Append the calculated y_final and theta_final values to the base arrays:
                base_y.append(y_final)
                base_theta.append(theta_final)

                # Updating y_prev and theta_prev:
                y_prev = y_final
                theta_prev = theta_final

                #Storing the calculated yi_final and yf_final values to the base arrays:
                base_yi.append(yi_final)
                base_yf.append(yf_final)

                #Updating yi_prev and yf_prev:
                yi_prev = yi_final
                yf_prev = yf_final



                if frame_count < forecast_steps: 

                    # If the frame count is less than the forecast_steps, use the full frame:
                    roi_min = 0
                    roi_max = frame_height
                    

                else: 

                    if frame_count == forecast_steps:
                        start_time = time.time() # Start the timer when the forecast_steps is reached

                    #Check if the GT horizon lines are within the ROI:
                    if yi_array_gt[frame_count] <= roi_max and yi_array_gt[frame_count] >= roi_min and yf_array_gt[frame_count] <= roi_max and yf_array_gt[frame_count] >= roi_min:
                        num_inclusions += 1
                    else:
                        num_exclusions += 1 # Increment the exclusion count if the GT horizon line is not within the ROI
                        print("Exclusion at frame:", frame_count, "roi_min and roi_max: ", roi_min, roi_max, "y_gt:", yi_array_gt[frame_count], yf_array_gt[frame_count])

                    if threshold != 0:
                        # Find the rolling mean of error array for the last forecast_steps elements excluding the last one:
                        rolling_mean = np.mean(error[-forecast_steps-1:-1])

                        # Calculate the rolling std of error array for the last forecast_steps elements excluding the last one:
                        rolling_std = np.std(error[-forecast_steps-1:-1])

                        # Calculate how many stds the last element is away from the rolling mean:
                        std_calc = abs(error[-1] - rolling_mean) / (rolling_std+1e-8)


                        if std_calc <= threshold:
                            # The z-score is below the threshold, so use ratio to calculate the new roi_min and roi_max:
                            y_min = min(yi_final, yf_final)
                            y_max = max(yi_final, yf_final)

                            roi_min = int(max(y_min - int(ratio * frame_height),0))
                            roi_max = int(min(y_max + int(ratio * frame_height),frame_height-1))

                        else:
                            # The z-score is above the threshold so use the full frame:
                            roi_min = 0
                            roi_max = frame_height
                    else: # If threshold is 0, use the full frame
                        roi_min = 0
                        roi_max = frame_height


                roi_diff_array.append(roi_max-roi_min)


                if save_output_video:
                    cv2.line(frame, (0, yi_final), (frame_width, yf_final), (0, 255, 255), 3)

                    # Use green color to plot the ground truth:
                    cv2.line(frame, (0, int(yi_array_gt[frame_count])), (frame_width, int(yf_array_gt[frame_count])), (0, 255, 0), 3)

                    # Use another color to draw the ROI:
                    cv2.line(frame, (0, roi_min), (frame_width, roi_min), (255, 0, 0), 2)
                    cv2.line(frame, (0, roi_max), (frame_width, roi_max), (255, 0, 0), 2)


                    # Write the frame to the output video
                    out.write(frame)

                frame_count += 1



            # Close the video file:
            cap.release()

            if save_output_video:
                out.release()

            #Finish timer:
            end_time = time.time()

            # Calculate the elapsed time:
            elapsed = end_time - start_time


            print(f"Elapsed time: {elapsed} seconds")
            print(f"Total HLDA time: {np.sum(HLDA_time_array)} seconds")

            

            # Create a new array called roi_diff_array_training_removed:
            roi_diff_array = np.array(roi_diff_array)
            roi_diff_array_training_removed = roi_diff_array[forecast_steps:]

            mean_real_err = np.mean(real_err_array)
            std_real_err = np.std(real_err_array)

            mean_roi_diff_training_removed = np.mean(roi_diff_array_training_removed)
            std_roi_diff_training_removed = np.std(roi_diff_array_training_removed)

            mean_roi_diff_training_removed_percent = 100*np.mean(roi_diff_array_training_removed)/frame_height

            std_roi_diff = np.std(roi_diff_array)

            mean_error = np.mean(error)
            std_error = np.std(error)

            total_time = np.sum(elapsed_time_array)
            mean_iter_array = np.mean(n_iter_array)

            # Append the values to the arrays:
            mean_real_err_array.append(mean_real_err)
            std_real_err_array.append(std_real_err)
            mean_roi_diff_training_removed_percent_array.append(mean_roi_diff_training_removed_percent)
            std_roi_diff_training_removed_array.append(std_roi_diff_training_removed)
            elapsed_array.append(elapsed)
            total_time_array.append(total_time)
            full_roi_count_array.append(full_roi_count)
            num_inclusions_array.append(num_inclusions)
            num_exclusions_array.append(num_exclusions)



            # Append the means to the CSV file:
            with open(csv_filename, 'a') as f:
                f.write(f"{dataset},{filename},{nowStr},{ratio},{threshold},{save_output_video},{full_roi_count},{mean_real_err},{std_real_err},{mean_roi_diff_training_removed_percent},{std_roi_diff_training_removed},{round(mean_error,2)},{round(std_error,2)},{round(elapsed,2)}, {round(total_time,2)}, {num_inclusions},{num_exclusions},{forecast_steps},{total_frames},{frame_width},{frame_height},{input_video_url}\n")

            # Printing the results:
            print(f"mean_real_err: {mean_real_err}, mean_roi_diff_training_removed_percent: {mean_roi_diff_training_removed_percent}, mean_error: {mean_error}, elapsed: {elapsed}")
            print(f"Full ROI count: {full_roi_count} out of {total_frames} frames")
            print("exiting...")

    with open(csv_filename, 'a') as f:
                f.write(f"Finished\n")

    # Append the means to the CSV file:
    with open(csv_filename_avg, 'a') as f:
        f.write(f"{dataset},{filename},{nowStr},{ratio},{threshold},{save_output_video},{np.mean(full_roi_count_array)},{np.mean(mean_real_err_array)},{np.mean(std_real_err_array)},{np.mean(mean_roi_diff_training_removed_percent_array)},{np.mean(std_roi_diff_training_removed_array)},{np.mean(elapsed_array)},{np.mean(total_time_array)},{np.mean(num_inclusions_array)},{np.mean(num_exclusions_array)},{forecast_steps},{frame_width},{frame_height}\n")
        

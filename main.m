% Hozion Line Detection and Tracking System - v2.0
% Dr Ahmet Agaoglu and Dr Nezih Topaloglu

% Choose dataset (Singapore-onboard or Buoy):
dataset = "Singapore-onboard";
%dataset = "Buoy";


% Parameters
ratio = 0.05;
threshold = 3;
forecast_steps = 20;


% Choose to save output video or not:
save_output_video = false;

% Set folder path based on operating system
if dataset == "Singapore-onboard"
    if ispc
        % Windows path - Update the path
        folder_files = fullfile('C:', 'User', 'Singapore-Onboard-Videos');
    else
        % Linux path - Update the path
        folder_files = fullfile('/', 'home', 'User', 'Singapore-Onboard-Videos');

    end
    
elseif dataset == "Buoy"
    if ispc
        % Windows path
        folder_files = fullfile('C:', 'User', 'Buoy-Videos');

    else
        % Linux path
        folder_files = fullfile('/', 'home', 'User', 'Buoy-Videos');

    end    

end

% Verify the folder exists
if ~exist(folder_files, 'dir')
    error(['Folder not found: ' folder_files]);
else
    disp(['Using video folder: ' folder_files]);
end


% CSV file path:
output_folder = fullfile(pwd, 'output');
csv_filename = fullfile(output_folder, 'output.csv');

% Check if file exist and write headers if it doesn't exist
if ~exist(csv_filename, 'file')
    fid = fopen(csv_filename, 'w');
    fprintf(fid, '%s\n', ...
        "Dataset,filename,frame_no, ratio,threshold,y,theta,roi_min,roi_max");
    fclose(fid);
end


    % Close any open figures or video objects
    close all;


    % Loop through all the files in the folder
    files = dir(fullfile(folder_files, '*.avi'));
    for file_idx = 1:length(files)
        
        
        file = files(file_idx).name;


        if endsWith(file, '.avi')
            input_video_url = fullfile(folder_files, file);
            [~, filename, ~] = fileparts(input_video_url);

            % Create output directories if they don't exist
            if ~exist('output', 'dir')
                mkdir('output');
            end
            if ~exist('output_videos', 'dir')
                mkdir('output_videos');
            end

            outputURL = fullfile('output_videos', [filename '_ratio_' num2str(ratio) '_thr_' num2str(threshold) '_onoff.mp4']);


            disp(outputURL)
            disp(input_video_url);

            % Video capture setup
            cap = VideoReader(input_video_url);
            frame_width = cap.Width;
            disp(['frame_width: ' num2str(frame_width)]);
            frame_height = cap.Height;
            disp(['frame_height: ' num2str(frame_height)]);

            if save_output_video
                out = VideoWriter(outputURL, 'Motion JPEG AVI');
                out.FrameRate = 30;
                open(out);
            end

            frame_count = 0;
            total_frames = floor(cap.Duration * cap.FrameRate);
            disp(['Total_frames: ' num2str(total_frames)]);

            % Initialize variables
            full_roi_count = 0;

            error = [];

            yi_prev = floor(frame_height / 2);
            yf_prev = floor(frame_height / 2);
            y_prev = floor(frame_height / 2);
            theta_prev = 0;

            roi_min = 0;
            roi_max = frame_height;

            x_ref = frame_height / 2;
            alpha = 0.1;

            t = 0:frame_width-1;

            H_true = zeros(total_frames, 2);

            % Create coordinate matrices
            [Xmat_or, Ymat_or] = meshgrid(1:frame_width, 1:frame_height);

            theta_limit_ultimate = rad2deg(atan(frame_height / frame_width));

            fid = fopen(csv_filename, 'a');

            % Main video processing loop
            while hasFrame(cap)

                roi_diff = roi_max - roi_min;
                if roi_diff == frame_height
                    full_roi_count = full_roi_count + 1;
                end

                theta_limit = rad2deg(atan(roi_diff / frame_width));
                LB = [0, -theta_limit];
                UB = [roi_diff, theta_limit];
                x_ref = frame_width / 2;

                % Read frame
                frame = readFrame(cap);

                if mod(frame_count, 100) == 0
                    disp(['Processing frame ' num2str(frame_count) '/' num2str(total_frames) ', threshold: ' num2str(threshold)]);
                end


                %*********************************************************
                %*********************************************************

                % Crop frame
                frame_to_process = frame(roi_min+1:roi_max, :, :); % MATLAB is 1-based

                % Convert to HSV
                HSV = rgb2hsv(frame);
                H = HSV(:, :, 1);
                S = HSV(:, :, 2);
                V = HSV(:, :, 3);


                % HL detection using thre HLDA
                res = HLDA(frame_to_process);
                y_final = res(1);
                theta_final = res(2);

  
                if y_final > 1e5
                    disp(['y_final is ' num2str(y_final) ', setting it and theta to previous value']);
                    y_final = y_prev;
                    theta_final = theta_prev;
                end

                [yi_final, yf_final] = convert_y_theta_to_yi_yf(y_final, theta_final, frame_width, true);

                yi_final = yi_final + roi_min;
                yf_final = yf_final + roi_min;
                y_final = y_final + roi_min;

 

                % Update H_true
                H_true(frame_count+1, :) = [y_final, theta_final];

                y = round(tan(deg2rad(H_true(frame_count+1, 2))) * t + H_true(frame_count+1, 1));

                if threshold ~= 0
                    error = [error; error_metric(...
                        H_true(frame_count+1, :)', x_ref, alpha, H, S, V, 3, Xmat_or, Ymat_or)];
                else
                    error = [error; 0];
                end

                if isnan(error(end))
                    error(end) = error(end-1);
                end



                y_prev = y_final;
                theta_prev = theta_final;

                yi_prev = yi_final;
                yf_prev = yf_final;

                % Display info at each 25 frames:
                if mod(frame_count, 25) == 0
                    disp([ 'Frame count: ' num2str(frame_count) ...
                        ', y and theta: ' num2str(y_final) ', ' num2str(theta_final) ...
                         ', yi and yf: ' num2str(yi_final) ', ' num2str(yf_final) ...
                         ]);

                end

                % Append to CSV file
                fprintf(fid, '%s,%s,%d,%.2f,%.1f,%.3f,%.3f,%.3f,%.3f\n', ...
                dataset, filename, frame_count, ratio, threshold, ...
                y_final, theta_final,...
                roi_min, roi_max);

                % If still in forecast_steps, do not estimate ROI:
                if frame_count < forecast_steps
                    roi_min = 0;
                    roi_max = frame_height;
                else
                    
                    % The threshold should be nonzero to estimate ROI:
                    if threshold ~= 0
                        % Calculate rolling_mean and rolling_std of error
                        % metric:
                        rolling_mean = mean(error(end-forecast_steps:end-1));
                        rolling_std = std(error(end-forecast_steps:end-1));
                        std_calc = abs(error(end) - rolling_mean) / (rolling_std + 1e-8);

                        if std_calc <= threshold
                            y_min = min(yi_final, yf_final);
                            y_max = max(yi_final, yf_final);

                            roi_min = max(y_min - floor(ratio * frame_height), 0);
                            roi_max = min(y_max + floor(ratio * frame_height), frame_height - 1);
                        else
                            % Z-score is above the threshold, use
                            % full-frame:
                            roi_min = 0;
                            roi_max = frame_height;
                        end
                    else
                        roi_min = 0;
                        roi_max = frame_height;
                    end

                end

                %**********************************************************
                %**********************************************************


		
                if save_output_video
                    % Draw lines on frame
                    frame = insertShape(frame, 'Line', [1 yi_final frame_width yf_final], ...
                                       'Color', [0 255 255], 'LineWidth', 3);
                    frame = insertShape(frame, 'Line', [1 roi_min frame_width roi_min], ...
                                       'Color', [255 0 0], 'LineWidth', 2);
                    frame = insertShape(frame, 'Line', [1 roi_max frame_width roi_max], ...
                                       'Color', [255 0 0], 'LineWidth', 2);

                    writeVideo(out, frame);
                end


                % Increment frame_count:
                frame_count = frame_count + 1;
            end

            % Clean up video objects
            if save_output_video
                close(out);
            end


            % Close the CSV file:
            fclose(fid);

            disp(['Full ROI count: ' num2str(full_roi_count) ' out of ' num2str(total_frames) ' frames']);
            disp('Video finished.');
        end


    
    end
	
% Write "Finished" to CSV
fid = fopen(csv_filename, 'a');
fprintf(fid, 'Finished\n');
fclose(fid);

		



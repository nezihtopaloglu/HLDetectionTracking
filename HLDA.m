 function x = HLDA(RGB)
        % HLDA: Detects the horizon line in a given video frame or a ROI.
        % Inputs:
        % RGB - The current video frame in RGB format.
        % Outputs:
        %   x - The estimated state of the horizon line [y_k, theta_k]
       
        % Convert the RGB image to grayscale and perform Canny edge detection
        [~, threshOut] = edge(rgb2gray(RGB), "canny");
       
        % Set the threshold to eliminate the weak edges
        thrNew = threshOut * 2;
        thrNew(thrNew >= 1) = 0.9999;
       
        % Apply Canny edge detection with the new threshold
        BW = edge(rgb2gray(RGB), "canny", thrNew);
       
        % Remove small objects from the binary image
        BW2 = bwareaopen(BW, round(sum(BW(:)) * 0.002));

        % Define the angles for Radon transform
        theta = 0:0.25:180 - 0.25;
       
        % Perform the Radon transform on the binary image
        [Rad, xp] = radon(BW2, theta);
       
        % Find the peak in the Radon transform which corresponds to the HL
        [row_peak, col_peak] = find(ismember(Rad, max(max(Rad))));
       
        % Get the distance and angle corresponding to the peak
        dist = xp(row_peak);
        th = theta(col_peak);
       
        % If multiple peaks, choose the first one
        dist = dist(1);
        th = th(1);
       
        % Calculate the orientation of the horizon line (theta_k)
        x(2) = 90 - th;
       
        % Calculate the vertical position of the horizon line (y_k)
        if dist ~= 0
            uy = sign(dist) * sind(th);
            A = abs(dist) / uy;
        else
            A = 0;
        end
        x(1) = -A + size(RGB, 1) / 2;
    end

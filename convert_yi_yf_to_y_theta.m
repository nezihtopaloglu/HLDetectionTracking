function [y, theta] = convert_yi_yf_to_y_theta(yi, yf, w, degree)
    % Convert the coordinates to the form (y, theta) where y is the vertical 
    % coordinate and theta is the angle (in degrees if degree=true)
    
    % Set default for degree if not provided
    if nargin < 4
        degree = false;
    end
    
    y = (yi + yf) / 2;
    theta = atan((yf - yi) / w);
    
    if degree
        theta = rad2deg(theta);
    end
    
    y = round(y); 
end

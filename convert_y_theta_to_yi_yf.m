
function [yi, yf] = convert_y_theta_to_yi_yf(y, theta, w, degree)
    % Convert the coordinates to the form (yi, yf) where yi and yf are the 
    % vertical coordinates, from (y, theta) representation
    
    % Set default for degree if not provided
    if nargin < 4
        degree = false;
    end
    
    if degree
        theta = deg2rad(theta);
    end
    
    yi = y - w * tan(theta)/2;
    yf = y + w * tan(theta)/2;
    
    yi = round(yi);  
    yf = round(yf);
end
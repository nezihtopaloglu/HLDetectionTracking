function f=error_metric(x,x_ref,alpha,I1,I2,I3,NofCol,Xmat,Ymat)

    yI = x(1); theta =x(2);
    c = yI-tand(theta)*x_ref;
    h_lim = 2*alpha*x_ref*cosd(theta);
    ind_ref = tand(theta)*Xmat+c;
    ind_lower = tand(theta)*Xmat+c-h_lim;
    ind_upper = tand(theta)*Xmat+c+h_lim;
    ind_sky = Ymat<ind_ref & Ymat>ind_lower;
    ind_sea = Ymat>ind_ref & Ymat<ind_upper;
    area_sky = sum(sum(ind_sky));
    area_sea = sum(sum(ind_sea));


    if NofCol>2
        n_sky = norm([var(I1(ind_sky)),var(I2(ind_sky)),var(I3(ind_sky))]);
        n_sea = norm([var(I1(ind_sea)),var(I2(ind_sea)),var(I3(ind_sea))]);
    elseif NofCol>1
        n_sky = norm([var(I1(ind_sky)),var(I2(ind_sky))]);
        n_sea = norm([var(I1(ind_sea)),var(I2(ind_sea))]);
    else
        n_sky = var(I1(ind_sky));
        n_sea = var(I1(ind_sea));
    end
    f = n_sky*area_sky/(area_sky+area_sea) + n_sea*area_sea/(area_sky+area_sea);


end
function s = getSatAlt(ILAT_degrees, alt_from_re_meters)
    RADS_PER_DEG = pi / 180.0; %rad per deg
    RADIUS_EARTH = 6.3712e6;   %meters
    
    L = RADIUS_EARTH / cos(ILAT_degrees * RADS_PER_DEG)^2;
    s_max = getSatLambda(ILAT_degrees, L);
    lambda = acos(sqrt((alt_from_re_meters + RADIUS_EARTH) / L)) / RADS_PER_DEG;
    
    s = s_max - getSatLambda(lambda, L);
end

function s = getSatLambda(lambda_deg, L)  %function is duplicated from getBFieldAtS.m
    RADS_PER_DEG = pi / 180;
    lambda = lambda_deg * RADS_PER_DEG;
    
    sinh_x = sqrt(3) * sin(lambda);
    x = log(sinh_x + sqrt(sinh_x.^2 + 1));
    
    s = (0.5 * L / sqrt(3)) * (x + 0.25 * (exp(2*x)-exp(-2*x)));
end
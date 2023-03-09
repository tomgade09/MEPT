function B = getBFieldAtS(s_meters)
    %getBFieldAtS Return B Field strength at s - the distance along the field line s
    ILAT = 72.0;
    B0 = 3.12e-5;
    RADS_PER_DEG = pi / 180;
    RADIUS_EARTH = 6371200;
    
    L = RADIUS_EARTH / cos(ILAT * RADS_PER_DEG)^2;
    L_norm = L / RADIUS_EARTH;
    s_max = getSatLambda(ILAT, L);
    
    lambda = getLambdaAtS(s_meters, s_max, ILAT, L);
    lam_rad = lambda * RADS_PER_DEG;
    rnorm = L_norm * cos(lam_rad).^2;
    
    B = -B0 / (rnorm.^3) * sqrt(1 + 3 * sin(lam_rad).^2);
end

function s = getSatLambda(lambda_deg, L)
    RADS_PER_DEG = pi / 180;
    lambda = lambda_deg * RADS_PER_DEG;
    
    sinh_x = sqrt(3) * sin(lambda);
    x = log(sinh_x + sqrt(sinh_x.^2 + 1));
    
    s = (0.5 * L / sqrt(3)) * (x + 0.25 * (exp(2*x)-exp(-2*x)));
end

function lambda_deg = getLambdaAtS(s, s_max, ILAT, L)
    lambdaEps = 1e-10;
    
    lambda_deg = (-ILAT/s_max) * s + ILAT;
    s_tmp = s_max - getSatLambda(lambda_deg, L);
    dlambda = 1;
    over = false;
    
    while (abs((s_tmp - s) / s) > lambdaEps)
        while (true)
            over = (s_tmp >= s);
            if (over)
                lambda_deg = lambda_deg + dlambda;
                s_tmp = s_max - getSatLambda(lambda_deg, L);
                if (s_tmp < s)
                    break;
                end
            else
                lambda_deg = lambda_deg - dlambda;
                s_tmp = s_max - getSatLambda(lambda_deg, L);
                if (s_tmp >= s)
                    break;
                end
            end
        end
        if (dlambda < lambdaEps / 100)
            break;
        end
        dlambda = dlambda / 5; 
    end
end
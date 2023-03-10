% This check is only valid for the simple mirror B case (no QSPS, Alfven wave, etc)

dataroot  = "../_dataout/230308_11.34.31.fastmath";
dataroot2 = "../_dataout/230308_13.12.13.nofastmath";
dataroot  = "../_dataout/230309_16.51.44.substantial_changes";
dataroot2 = "../_dataout/230309_18.51.33";
%dataroot2 = "../_dataout/230309_17.06.12.doubles";
%dataroot  = "../_dataout/230309_18.45.10.dbl.O2.RK4";


ILAT = 72.0;
RADIUS_EARTH = 6.3712e6;          %meters
s_ion = getSatAlt(ILAT, 620e3);   %alt 620 km
s_sat = getSatAlt(ILAT, 4000e3);  %alt 4000 km
s_mag = 19881647.2473464;         %to match c++ code exactly
B_ion = getBFieldAtS(s_ion);
B_sat = getBFieldAtS(s_sat);
B_mag = getBFieldAtS(s_mag);

%make sure the 'float' or 'double' option matches the datatype output by the simulation
%or you will get errors
bins_fm = loaddatafold(dataroot,'float');
bins_no = loaddatafold(dataroot2,'float');


[E_fm_max_err, PA_fm_max_err, E_fm_RMSE, PA_fm_RMSE, E_fm_min_err, PA_fm_min_err] = ...
    checkError(bins_fm.satellites.a4e6ElecDng_index, ...
               bins_fm.satellites.a4e6ElecDng_E, bins_fm.satellites.a4e6ElecDng_PA, ...
               bins_fm.init.elec_E, bins_fm.init.elec_PA, bins_fm.init.elec_s, ...
               B_mag, B_sat, B_ion, B_sat);
           
[E_no_max_err, PA_no_max_err, E_no_RMSE, PA_no_RMSE, E_no_min_err, PA_no_min_err] = ...
    checkError(bins_no.satellites.a4e6ElecDng_index, ...
               bins_no.satellites.a4e6ElecDng_E, bins_no.satellites.a4e6ElecDng_PA, ...
               bins_no.init.elec_E, bins_no.init.elec_PA, bins_no.init.elec_s, ...
               B_mag, B_sat, B_ion, B_sat);


function [E_max_err, PA_max_err, E_RMSE, PA_RMSE, E_min_err, PA_min_err] = ...
               checkError(particle_index, E_chk, PA_chk, E_init, PA_init, s_init, ...
                          B_dn_init, B_dn_final, B_up_init, B_up_final)
    top_mask = (s_init > 1.0e6);  %identifies particles starting at top of sim (not bottom)
    PA_final = zeros(size(PA_init));
    PA_final(top_mask)  = mirrorPA(PA_init(top_mask),  B_dn_init, B_dn_final);
    PA_final(~top_mask) = mirrorPA(PA_init(~top_mask), B_up_init, B_up_final);
    
    err = @(base,comp) abs((base-comp)./base);
    RMSE = @(base,comp,num) (sum((base-comp).^2))/num;
    
    particle_index = particle_index(particle_index <= 3455999) + 1;
    sz = size(particle_index);
    basemask = full(sparse(particle_index, ones(sz), true(sz), 3456000, 1));
    compmask = false(size(E_chk));    %mask off values with an index > 3456000
    compmask(1:size(particle_index,1)) = true;
    compmask(PA_final(basemask) == -1) = false;
    basemask(PA_final == -1) = false;
    
    E_err  = err(E_init(basemask),E_chk(compmask));
    PA_err = err(PA_final(basemask),PA_chk(compmask));
    E_max_err = max(E_err);
    E_min_err = min(E_err);
    PA_max_err = max(PA_err);
    PA_min_err = min(PA_err);
    E_RMSE = RMSE(E_init(basemask),E_chk(compmask),sum(basemask));
    PA_RMSE = RMSE(PA_final(basemask),PA_chk(compmask),sum(basemask));
end


function PAf_deg = mirrorPA(PA0_deg, Binit, Bfinal)
    % Mirrors an initial pitch angle due to the B field at Binit to Bfinal (due to mirror force only)
    % Produces similar values to those produced by MEPT - time-stepping particle dynamics code
    
    if ((any(PA0_deg < 90.0) && abs(Binit) > abs(Bfinal)) ||...
        (any(PA0_deg > 90.0) && abs(Binit) < abs(Bfinal)))
        error("mirrorPA: particle, Bfield gradient mismatch");
    end
    if (Binit == Bfinal)
        PAf_deg = PA0_deg;
        return;
    end
    
    %intermediate value for computing final angle
    PAf_deg = zeros(size(PA0_deg));
    PAf_deg(PA0_deg >=0) = Binit / Bfinal * (1 + 1./(tan_DEG(PA0_deg(PA0_deg >=0)).^2)) - 1;
    PAf_deg(PA0_deg < 0) = PA0_deg(PA0_deg < 0);
    
    PAf_deg(PAf_deg < 0) = -1; %if this is the case, particle reflects before Bfinal
    PAf_deg(PAf_deg >=0) = atan_DEG(sqrt(1./PAf_deg(PAf_deg >=0)));
    if (abs(Binit) > abs(Bfinal))  %if B field strength drops off, no mirroring occurs
        PAf_deg = 180 - PAf_deg;   %PA > 90 is defined as upgoing (away from Earth)
    end
end

function ret = tan_DEG(deg)
    RADS_PER_DEG = pi / 180;
    ret = tan(deg * RADS_PER_DEG);
end

function deg = atan_DEG(num)
    RADS_PER_DEG = pi / 180;
    deg = atan(num) / RADS_PER_DEG;
end

function [E, PA] = v2DtoEPitch(vpara, vperp)
    me = 9.10938356e-31;
    J_per_eV = 1.6021766209e-19;
    rads_per_deg = pi / 180.0;
    
    E = 0.5 * me / J_per_eV * (vpara .* vpara + vperp .* vperp);
    PA = atan2(vperp, -vpara) / rads_per_deg;
end

function data = loaddatafold(rootfold, prec)
    rootfold = rootfold + "/bins/";
    
    data = struct();
    data.init = loadfold(rootfold + "particles_init/", prec);
    data.final = loadfold(rootfold + "particles_final/", prec);
    data.satellites = loadfold(rootfold + "satellites/", prec);
end

function bins = loadfold(fold, prec)
    flist = dir(fold);
    bins = struct();
    
    setlist = [];
    
    for (i = 1:size(flist,1))
        wherebin = strfind(flist(i).name, '.bin');
        if (~isempty(wherebin))
            nm = flist(i).name(1:wherebin-1);
            if (~isletter(nm(1)))
                nm = strcat('a',nm);
            end
            if (~isempty(strfind(flist(i).name, '_index.bin')))
                bins.(nm) = loadbin(fold + flist(i).name, 'uint32');
                setlist = [setlist, string(nm(1:end-5))];
            else
                bins.(nm) = loadbin(fold + flist(i).name, prec);
                if (max(size(bins.(nm))) ~= 3456000) %relies on '_index' being the first file of set
                    mask = (bins.(setlist(end)+"index") <= 3455999);
                    S = sparse(bins.(setlist(end)+"index")(mask)+1, ones(1,sum(mask)), ...%row, col
                               bins.(nm)(mask),3456000,1);  %sparse data, final row, col count
                    bins.(nm+"_exp") = full(S);
                end
            end
        end
    end
    
    if (isempty(setlist))
        loc = strfind(flist(3).name, '_s.bin');
        prefix = flist(3).name(1:loc);
        [bins.(prefix + "E"), bins.(prefix + "PA")] = ...
            v2DtoEPitch(bins.(prefix + "vpara"), bins.(prefix + "vperp"));
    end
    
    for (set = 1:max(size(setlist)))
        [bins.(setlist(set) + "E"), bins.(setlist(set) + "PA")] = ...
            v2DtoEPitch(bins.(setlist(set) + "vpara"), bins.(setlist(set) + "vperp"));
    end
end

function bin = loadbin(fname,prec)
    fID = fopen(fname,'r');
    bin = fread(fID,prec);
    fclose(fID);
end
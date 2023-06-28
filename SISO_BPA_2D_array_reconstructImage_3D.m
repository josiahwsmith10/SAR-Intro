function bpa = SISO_BPA_2D_array_reconstructImage_3D(sar,fmcw,bpa)
% sarData is of size (sar.Nx, sar.Ny, fmcw.Nk)
k = single(reshape(fmcw.k,1,1,[]));

%% Get Antenna Positions (primed)
xp_m = reshape(single(sar.x_m),[],1,1);
yp_m = reshape(single(sar.y_m),1,[],1);
zp_m = reshape(single(sar.z_m),1,1,[]);

%% Get Scene Positions (unprimed)
x_m = reshape(single(bpa.x_m),1,1,1,[],1,1);
y_m = reshape(single(bpa.y_m),1,1,1,1,[],1);
z_m = reshape(single(bpa.z_m),1,1,1,1,1,[]);

sarImage = single(zeros(length(bpa.x_m),length(bpa.y_m),length(bpa.z_m)));
sarData = single(sar.sarData);

%% Use gpuArray if Possible
try
    gpuArray(1);
    isGPU = true;
catch
    isGPU = false;
    warning("No GPU - Expect Excessively Long Computation Times")
end
if isGPU
    acceptableX = length(bpa.x_m);
    while true
        reset(gpuDevice);
        try
            R = gpuArray(sqrt( (x_m(1:acceptableX) - xp_m).^2 + (y_m(1) - yp_m).^2 + (z_m(1) - zp_m).^2));
            bpaKernel = R.^2 .* exp(-1j*k*2.*R);
            a = sum(gpuArray(sarData) .* bpaKernel,[1,2,3]);
        catch
            acceptableX = acceptableX - 1;
            if acceptableX >= length(bpa.x_m)/2
                continue;
            else
                acceptableX = 1;
            end
        end
        break;
    end
    reset(gpuDevice);
end

%% Compute BPA (VERY LONG!)
d = waitbar(0,"Computing BPA, Please Wait!");
count = 0;

numX = length(bpa.x_m);
numY = length(bpa.y_m);
numZ = length(bpa.z_m);
if isGPU
    for indX = 1:acceptableX:numX
        for indY = 1:numY
            for indZ = 1:numZ
%                 reset(gpuDevice);
                R = gpuArray(sqrt( (x_m(indX:min([indX+acceptableX-1,numX])) - xp_m).^2 + (y_m(indY) - yp_m).^2 + (z_m(indZ) - zp_m).^2));
                bpaKernel = R.^2 .* exp(-1j*k*2.*R);
                sarImage(indX:min([indX+acceptableX-1,numX]),indY,indZ) = gather(sum(gpuArray(sarData) .* bpaKernel,[1,2,3]));
                
                count = count + 1;
                waitbar(count/numel(sarImage)*acceptableX,d,"Computing BPA, Please Wait!");
            end
        end
    end
else
    for indX = 1:numX
        for indY = 1:numY
            for indZ = 1:numZ
                R = gpuArray(sqrt( (x_m(indX) - xp_m).^2 + (y_m(indY) - yp_m).^2 + (z_m(indZ) - zp_m).^2));
                bpaKernel = R.^2 .* exp(-1j*k*2.*R);
                sarImage(indX,indY,indZ) = sum(sarData .* bpaKernel,'all');
                
                count = count + 1;
                waitbar(count/numel(sarImage),d,"Computing BPA, Please Wait!");
            end
        end
    end
end

delete(d)

bpa.pxyz = gather(abs(sarImage));

if isGPU
    reset(gpuDevice);
end
end
%% 参数设置
clear; clc;

Nx = 301; Ny = 101; Nz = 21;       % 与 C++ 里的 fluidSize 一致
Ncell = Nx * Ny * Nz;

zIndex = 10;                        % 取中间层 xy 剖面，可改
skip = 3;                           % quiver 下采样步长，避免太密

% 根据你输出的时间步来设置，这里假设从 0 到 100，每步 1
timeList = 0:1:100;                 % <- 根据你实际的 time 修改

%% 创建视频
v = VideoWriter('xy_slice.avi', 'Motion JPEG AVI');
v.FrameRate = 10;                   % 帧率，可改
open(v);

figure('Position',[100 100 900 700]);

[X, Y] = meshgrid(1:Nx, 1:Ny);      % 网格，用于 quiver

for t = timeList
    velFile  = sprintf('Velb%d.dat', t);
    concFile = sprintf('Conc%d.dat', t);

    % 如果某个时间步没输出文件，就跳过
    if ~isfile(velFile) || ~isfile(concFile)
        warning('t = %d: 文件缺失，跳过', t);
        continue;
    end

    %% 读速度数据 Velb*.dat
    % 文件中每行: ux  uy  uz
    fid = fopen(velFile, 'r');
    velData = fscanf(fid, '%f', [3, Ncell])';   % 每行三个数
    fclose(fid);

    ux = reshape(velData(:,1), [Nx, Ny, Nz]);
    uy = reshape(velData(:,2), [Nx, Ny, Nz]);
    uz = reshape(velData(:,3), [Nx, Ny, Nz]);   %#ok<NASGU> % 如需 z 分量可用

    %% 读浓度数据 Conc*.dat
    fid = fopen(concFile, 'r');
    concData = fscanf(fid, '%f', [1, Ncell])';
    fclose(fid);

    c = reshape(concData, [Nx, Ny, Nz]);

    %% 取固定 z 层的 xy 剖面
    uxSlice = ux(:,:,zIndex);
    uySlice = uy(:,:,zIndex);
    cSlice  = c(:,:,zIndex);

    %% 绘图：背景为浓度场，叠加速度矢量
    clf;
    % 注意转置和 YDir，保证 x 轴向右，y 轴向上
    imagesc(1:Nx, 1:Ny, cSlice');   
    set(gca, 'YDir', 'normal');
    axis equal tight;
    colormap(jet);
    colorbar;
    clim([0 0.2]);
    hold on;

    % 下采样后画矢量场
    xs = 1:skip:Nx;
    ys = 1:skip:Ny;
    quiver(X(ys,xs), Y(ys,xs), ...
           uxSlice(xs,ys)', uySlice(xs,ys)', 2, 'k'); % 2 为缩放系数，可调

    title(sprintf('XY剖面 z = %d, time = %d', zIndex, t), 'FontSize', 14);
    xlabel('x'); ylabel('y');

    drawnow;
    frame = getframe(gcf);
    writeVideo(v, frame);
end

close(v);
disp('视频 xy_slice.avi 已生成');

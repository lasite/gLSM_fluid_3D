%% ================== gel_speed_stream_3views_2gels_smooth.m ==================
% Method: vector polygon patches (iso-contour -> patch), smoothed boundary; record videos
% Display: three slices (XY / YZ / XZ): speed map + 2D streamlines + gray solid (vector boundary)
% Adapted for TWO GELS based on reference dimensions.
clear; clc; close all;

%% Paths & frames
data_dir   = 'D:\repos\gLSM_fluid_3D';

% --- Modified filenames for 2 Gels ---
fmt_rm1    = 'gel1rm%d.dat';
fmt_rm2    = 'gel2rm%d.dat';
fmt_velb   = 'Velb%d.dat';

frames     = 0:1:100;

%% Grid sizes (from reference script)
nx_s1=15; ny_s1=15; nz_s1=4;      % Solid 1
nx_s2=25; ny_s2=25; nz_s2=4;      % Solid 2
nx_f=301; ny_f=101; nz_f=21;      % Fluid

%% Grid steps & offsets
h_s = 1.0;
h_f = 0.5;
scale_f2s = h_f / h_s;                  % fluid index -> physical coord
use_cell_center_vel = true;
offset_f = (use_cell_center_vel) * (0.5 * scale_f2s);

%% Visualization params
cmap_name        = 'parula';
stream_density   = 1.2;
stream_lw        = 1.05;
fix_speed_clim   = true;                % unified color scale
speed_clim       = [0, 0.001];          % Adjust based on your data range
perc_for_clim    = 99;
gray_val         = 0.88;                % solid gray
boundary_color   = [0.12 0.12 0.12];
boundary_lw      = 1.6;

%% Boundary extraction & smoothing
upsample         = 6;       % slice supersampling factor
sigma_edge       = 1.2;     % Gaussian smoothing before contouring; =0 disables

%% alphaShape params
use_alpha_override = true;
alpha_value        = 1.5;   % Radius for alpha shape (tightness)

%% Slice indices (defaults to mid planes)
ix_plane = round((nx_f+1)/2);   % YZ (x = const)
iy_plane = round((ny_f+1)/2);   % XZ (y = const)
iz_plane = round((nz_f+1)/2);   % XY (z = const)

%% Export settings: 'combined' (single video) or 'separate' (three videos)
export_mode = 'separate';                 % 'combined' or 'separate'
fps         = 10;                         % from reference
vid_name_combined = fullfile(data_dir, 'gel_3views_2gels.mp4');
vid_name_xy = fullfile(data_dir, 'gel_xy_2gels.mp4');
vid_name_yz = fullfile(data_dir, 'gel_yz_2gels.mp4');
vid_name_xz = fullfile(data_dir, 'gel_xz_2gels.mp4');

%% Small utilities
toYXZ = @(v,nx,ny,nz) permute(reshape(v,[nx,ny,nz]),[2 1 3]);  % (ny,nx,nz)

%% Physical coords (fluid grid)
xF = (0:nx_f-1) * scale_f2s + offset_f;
yF = (0:ny_f-1) * scale_f2s + offset_f;
zF = (0:nz_f-1) * scale_f2s + offset_f;

%% Figures / videos setup
switch export_mode
    case 'combined'
        fig = figure('Color','w','Units','pixels','Position',[40 40 1500 520], ...
                     'Renderer','opengl','GraphicsSmoothing','on');
        tlo = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

        ax_xy = nexttile(tlo,1); xlabel(ax_xy,'X','FontSize',20); ylabel(ax_xy,'Y','FontSize',20);
        title(ax_xy,'XY','FontSize',20,'Interpreter','none'); grid(ax_xy,'on'); box(ax_xy,'on'); set(ax_xy,'FontSize',20);

        ax_yz = nexttile(tlo,2); xlabel(ax_yz,'Z','FontSize',20); ylabel(ax_yz,'Y','FontSize',20);
        title(ax_yz,'YZ','FontSize',20,'Interpreter','none'); grid(ax_yz,'on'); box(ax_yz,'on'); set(ax_yz,'FontSize',20);

        ax_xz = nexttile(tlo,3); xlabel(ax_xz,'X','FontSize',20); ylabel(ax_xz,'Z','FontSize',20);
        title(ax_xz,'XZ','FontSize',20,'Interpreter','none'); grid(ax_xz,'on'); box(ax_xz,'on'); set(ax_xz,'FontSize',20);

        colormap(fig, cmap_name);
        cb1 = colorbar('peer', ax_xy); cb1.Label.String = '$|\mathbf{V}|$'; cb1.Label.Interpreter = 'latex'; cb1.Label.FontSize = 20; set(cb1,'FontSize',20);
        cb2 = colorbar('peer', ax_yz); cb2.Label.String = '$|\mathbf{V}|$'; cb2.Label.Interpreter = 'latex'; cb2.Label.FontSize = 20; set(cb2,'FontSize',20);
        cb3 = colorbar('peer', ax_xz); cb3.Label.String = '$|\mathbf{V}|$'; cb3.Label.Interpreter = 'latex'; cb3.Label.FontSize = 20; set(cb3,'FontSize',20);

        try, vw = VideoWriter(vid_name_combined,'MPEG-4'); catch, vw = VideoWriter(strrep(vid_name_combined,'.mp4','.avi'),'Motion JPEG AVI'); end
        vw.FrameRate = fps; open(vw);

    case 'separate'
        fig_xy = figure('Color','w','Units','pixels','Position',[60 40 800 400],'Renderer','opengl','GraphicsSmoothing','on');
        ax_xy  = axes('Parent',fig_xy); colormap(fig_xy, cmap_name);
        xlabel(ax_xy,'X','FontSize',20); ylabel(ax_xy,'Y','FontSize',20);
        title(ax_xy,'XY','FontSize',20,'Interpreter','none'); grid(ax_xy,'on'); box(ax_xy,'on'); set(ax_xy,'FontSize',20);
        cb1 = colorbar('peer',ax_xy);

        fig_yz = figure('Color','w','Units','pixels','Position',[60 40 600 550],'Renderer','opengl','GraphicsSmoothing','on');
        ax_yz  = axes('Parent',fig_yz); colormap(fig_yz, cmap_name);
        xlabel(ax_yz,'Z','FontSize',20); ylabel(ax_yz,'Y','FontSize',20);
        title(ax_yz,'YZ','FontSize',20,'Interpreter','none'); grid(ax_yz,'on'); box(ax_yz,'on'); set(ax_yz,'FontSize',20);
        cb2 = colorbar('peer',ax_yz);

        fig_xz = figure('Color','w','Units','pixels','Position',[1080 40 500 480],'Renderer','opengl','GraphicsSmoothing','on');
        ax_xz  = axes('Parent',fig_xz); colormap(fig_xz, cmap_name);
        xlabel(ax_xz,'X','FontSize',20); ylabel(ax_xz,'Z','FontSize',20);
        title(ax_xz,'XZ','FontSize',20,'Interpreter','none'); grid(ax_xz,'on'); box(ax_xz,'on'); set(ax_xz,'FontSize',20);
        cb3 = colorbar('peer',ax_xz);

        try, vw_xy = VideoWriter(vid_name_xy,'MPEG-4'); catch, vw_xy=VideoWriter(strrep(vid_name_xy,'.mp4','.avi'),'Motion JPEG AVI'); end
        try, vw_yz = VideoWriter(vid_name_yz,'MPEG-4'); catch, vw_yz=VideoWriter(strrep(vid_name_yz,'.mp4','.avi'),'Motion JPEG AVI'); end
        try, vw_xz = VideoWriter(vid_name_xz,'MPEG-4'); catch, vw_xz=VideoWriter(strrep(vid_name_xz,'.mp4','.avi'),'Motion JPEG AVI'); end
        [vw_xy.FrameRate, vw_yz.FrameRate, vw_xz.FrameRate] = deal(fps);
        open(vw_xy); open(vw_yz); open(vw_xz);
end

for t = frames
    %% Read files
    file_rm1 = fullfile(data_dir, sprintf(fmt_rm1, t));
    file_rm2 = fullfile(data_dir, sprintf(fmt_rm2, t));
    file_vb  = fullfile(data_dir, sprintf(fmt_velb, t));

    if ~(exist(file_rm1,'file') && exist(file_rm2,'file') && exist(file_vb,'file'))
        fprintf('[skip] Missing frame %d files\n', t); continue;
    end

    % --- Load Gel 1 ---
    RM1 = load(file_rm1);
    if size(RM1,1) ~= nx_s1*ny_s1*nz_s1
        warning('Frame %d: rm1 size mismatch, skip.', t); continue;
    end
    Xs1 = reshape(RM1(:,1), [nx_s1,ny_s1,nz_s1]);
    Ys1 = reshape(RM1(:,2), [nx_s1,ny_s1,nz_s1]);
    Zs1 = reshape(RM1(:,3), [nx_s1,ny_s1,nz_s1]);

    % --- Load Gel 2 ---
    RM2 = load(file_rm2);
    if size(RM2,1) ~= nx_s2*ny_s2*nz_s2
        warning('Frame %d: rm2 size mismatch, skip.', t); continue;
    end
    Xs2 = reshape(RM2(:,1), [nx_s2,ny_s2,nz_s2]);
    Ys2 = reshape(RM2(:,2), [nx_s2,ny_s2,nz_s2]);
    Zs2 = reshape(RM2(:,3), [nx_s2,ny_s2,nz_s2]);

    % --- Load Fluid ---
    VB = load(file_vb);
    if size(VB,1) ~= nx_f*ny_f*nz_f
        warning('Frame %d: Velb size mismatch, skip.', t); continue;
    end
    Uf = toYXZ(VB(:,1), nx_f, ny_f, nz_f);
    Vf = toYXZ(VB(:,2), nx_f, ny_f, nz_f);
    Wf = toYXZ(VB(:,3), nx_f, ny_f, nz_f);
    speed = sqrt(Uf.^2 + Vf.^2 + Wf.^2);

    %% alphaShape (solid volumes)
    % Gel 1
    try
        shp1 = alphaShape(Xs1(:), Ys1(:), Zs1(:));
        if use_alpha_override, shp1.Alpha = alpha_value; end
    catch
        shp1 = [];
    end
    % Gel 2
    try
        shp2 = alphaShape(Xs2(:), Ys2(:), Zs2(:));
        if use_alpha_override, shp2.Alpha = alpha_value; end
    catch
        shp2 = [];
    end

    %% Color scale
    if fix_speed_clim
        clim_now = speed_clim;
    else
        if exist('prctile','file'), smax = prctile(speed(:), perc_for_clim);
        else, smax = max(speed(:)); end
        clim_now = [0 max(eps, smax)];
    end

    %% Base grids (original resolution)
    [XX_xy, YY_xy] = meshgrid(xF, yF);
    [ZZ_yz, YY_yz] = meshgrid(zF, yF);
    [XX_xz, ZZ_xz] = meshgrid(xF, zF);

    %% ================== 1) XY (z = const) ==================
    S_xy = speed(:,:,iz_plane);
    U_xy = Uf(:,:,iz_plane);
    V_xy = Vf(:,:,iz_plane);

    cla(ax_xy); hold(ax_xy,'on');
    imagesc(ax_xy, [min(xF) max(xF)], [min(yF) max(yF)], S_xy);
    set(ax_xy,'YDir','normal'); axis(ax_xy,'image'); caxis(ax_xy,clim_now);

    % Extract polys from BOTH shapes
    polys1 = slice_polys_from_alpha(shp1, 'xy', zF(iz_plane), xF, yF, upsample, sigma_edge);
    polys2 = slice_polys_from_alpha(shp2, 'xy', zF(iz_plane), xF, yF, upsample, sigma_edge);
    polys_all = [polys1, polys2]; % Combine

    % Mask and Streamlines
    mask_xy = mask_from_polys(polys_all, XX_xy, YY_xy);
    U_plot = U_xy; V_plot = V_xy; U_plot(mask_xy)=NaN; V_plot(mask_xy)=NaN;
    hxy = streamslice(ax_xy, XX_xy, YY_xy, U_plot, V_plot, stream_density);
    set(hxy,'Color','k','LineWidth',stream_lw);

    % Draw patches
    for k = 1:numel(polys_all)
        px = polys_all{k}(1,:); py = polys_all{k}(2,:);
        patch('XData',px,'YData',py, ...
              'FaceColor',gray_val*[1 1 1], 'EdgeColor',boundary_color, ...
              'LineWidth',boundary_lw, 'FaceAlpha',1.0, 'HitTest','off', ...
              'Parent',ax_xy);
    end
    title(ax_xy, sprintf('Time = %d', t), 'FontSize',20,'Interpreter','none');
    hold(ax_xy,'off');

    %% ================== 2) YZ (x = const) ==================
    S_yz = squeeze(speed(:, ix_plane, :));   % (ny, nz)
    U_yz = squeeze(Wf(:, ix_plane, :));      % horizontal Z
    V_yz = squeeze(Vf(:, ix_plane, :));      % vertical Y

    cla(ax_yz); hold(ax_yz,'on');
    imagesc(ax_yz, [min(zF) max(zF)], [min(yF) max(yF)], S_yz);
    set(ax_yz,'YDir','normal'); axis(ax_yz,'image'); caxis(ax_yz,clim_now);

    polys1 = slice_polys_from_alpha(shp1, 'yz', xF(ix_plane), zF, yF, upsample, sigma_edge);
    polys2 = slice_polys_from_alpha(shp2, 'yz', xF(ix_plane), zF, yF, upsample, sigma_edge);
    polys_all = [polys1, polys2];

    mask_yz = mask_from_polys(polys_all, ZZ_yz, YY_yz);
    U_plot = U_yz; V_plot = V_yz; U_plot(mask_yz)=NaN; V_plot(mask_yz)=NaN;
    hyz = streamslice(ax_yz, ZZ_yz, YY_yz, U_plot, V_plot, stream_density);
    set(hyz,'Color','k','LineWidth',stream_lw);

    for k = 1:numel(polys_all)
        pz = polys_all{k}(1,:); py = polys_all{k}(2,:);
        patch('XData',pz,'YData',py, ...
              'FaceColor',gray_val*[1 1 1], 'EdgeColor',boundary_color, ...
              'LineWidth',boundary_lw, 'FaceAlpha',1.0, 'HitTest','off', ...
              'Parent',ax_yz);
    end
    title(ax_yz, sprintf('Time = %d', t), 'FontSize',20,'Interpreter','none');
    hold(ax_yz,'off');

    %% ================== 3) XZ (y = const) ==================
    S_xz = squeeze(speed(iy_plane, :, :)).';  % (nz, nx)
    U_xz = squeeze(Uf(iy_plane, :, :)).';     % X
    V_xz = squeeze(Wf(iy_plane, :, :)).';     % Z

    cla(ax_xz); hold(ax_xz,'on');
    imagesc(ax_xz, [min(xF) max(xF)], [min(zF) max(zF)], S_xz);
    set(ax_xz,'YDir','normal'); axis(ax_xz,'image'); caxis(ax_xz,clim_now);

    polys1 = slice_polys_from_alpha(shp1, 'xz', yF(iy_plane), xF, zF, upsample, sigma_edge);
    polys2 = slice_polys_from_alpha(shp2, 'xz', yF(iy_plane), xF, zF, upsample, sigma_edge);
    polys_all = [polys1, polys2];

    mask_xz = mask_from_polys(polys_all, XX_xz, ZZ_xz);
    U_plot = U_xz; V_plot = V_xz; U_plot(mask_xz)=NaN; V_plot(mask_xz)=NaN;
    hxz = streamslice(ax_xz, XX_xz, ZZ_xz, U_plot, V_plot, stream_density);
    set(hxz,'Color','k','LineWidth',stream_lw);

    for k = 1:numel(polys_all)
        px = polys_all{k}(1,:); pz = polys_all{k}(2,:);
        patch('XData',px,'YData',pz, ...
              'FaceColor',gray_val*[1 1 1], 'EdgeColor',boundary_color, ...
              'LineWidth',boundary_lw, 'FaceAlpha',1.0, 'HitTest','off', ...
              'Parent',ax_xz);
    end
    title(ax_xz, sprintf('Time = %d', t), 'FontSize',20,'Interpreter','none');
    hold(ax_xz,'off');

    %% Write video frame(s)
    switch export_mode
        case 'combined'
            drawnow;
            writeVideo(vw, getframe(fig));
        case 'separate'
            drawnow;
            writeVideo(vw_xy, getframe(fig_xy));
            writeVideo(vw_yz, getframe(fig_yz));
            writeVideo(vw_xz, getframe(fig_xz));
    end
end

%% Close videos
switch export_mode
    case 'combined'
        close(vw);
        fprintf('Exported video: %s\n', vid_name_combined);
    case 'separate'
        close(vw_xy); close(vw_yz); close(vw_xz);
        fprintf('Exported videos:\n  %s\n  %s\n  %s\n', vid_name_xy, vid_name_yz, vid_name_xz);
end

%% ================== Local helper functions ==================
function polys = slice_polys_from_alpha(shp, plane, val, ax1, ax2, upsample, sigma_edge)
% On a given slice (plane: 'xy'/'yz'/'xz', val is the constant coord):
% 1) build a high-res grid; 2) inShape to get binary mask; 3) optional Gaussian smoothing;
% 4) contourc(...,[0.5 0.5]) to extract iso-contours -> polygons (2xN vertices)
    polys = {};
    if isempty(shp), return; end
    ax1_hr = linspace(min(ax1), max(ax1), max(8, upsample*numel(ax1)));
    ax2_hr = linspace(min(ax2), max(ax2), max(8, upsample*numel(ax2)));
    [A1, A2] = meshgrid(ax1_hr, ax2_hr);
    switch lower(plane)
        case 'xy'
            A3 = val * ones(size(A1));                     % z = const
            mask = inShape(shp, A1, A2, A3);
            M = double(mask);
        case 'yz'
            A3 = val * ones(size(A1));                     % x = const; horiz=Z, vert=Y
            mask = inShape(shp, A3, A2, A1);               % (x,y,z)=(const, Y, Z)
            M = double(mask);
        case 'xz'
            A3 = val * ones(size(A1));                     % y = const; horiz=X, vert=Z
            mask = inShape(shp, A1, A3, A2);               % (x,y,z)=(X, const, Z)
            M = double(mask);
        otherwise
            error('plane must be xy/yz/xz');
    end
    if sigma_edge > 0
        M = gauss_smooth2(M, sigma_edge);
    end
    C = contourc(ax1_hr, ax2_hr, M, [0.5 0.5]);           % ContourMatrix
    polys = contourc_to_polys(C);
end

function polys = contourc_to_polys(C)
% Parse contourc ContourMatrix into {2xN, ...} vertex lists
    polys = {};
    i = 1;
    while i < size(C,2)
        npts  = C(2,i);
        verts = C(:, i+1:i+npts);
        i = i + npts + 1;
        if npts >= 3
            polys{end+1} = verts; %#ok<AGROW>
        end
    end
end

function mask = mask_from_polys(polys, XX, YY)
% Build a union mask from polygons (for streamlines masking)
    mask = false(size(XX));
    for k = 1:numel(polys)
        px = polys{k}(1,:); py = polys{k}(2,:);
        mask = mask | inpolygon(XX, YY, px, py);
    end
end

function A = gauss_smooth2(A, sigma)
% Simple 2D Gaussian smoothing (falls back to conv if imgaussfilt is absent)
    if sigma<=0, return; end
    if exist('imgaussfilt','file')
        A = imgaussfilt(A, sigma);
    else
        r = max(1, ceil(3*sigma));
        x = (-r:r);
        g = exp(-(x.^2)/(2*sigma^2)); g = g/sum(g);
        A = conv2(conv2(A, g, 'same'), g', 'same');
    end
end
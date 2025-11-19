%% ================== gel_faces_quiver_only_2gels.m ==================
clear; clc;

%% 路径与帧
data_dir   = 'D:\repos\gLSM_fluid_3D';

% ---------- 修改：两个凝胶 ----------
fmt_rm1    = 'gel1rm%d.dat';   % 凝胶1
fmt_vm1    = 'gel1vm%d.dat';
fmt_rm2    = 'gel2rm%d.dat';   % 凝胶2
fmt_vm2    = 'gel2vm%d.dat';
% -----------------------------------

fmt_velb   = 'Velb%d.dat';
frames     = 0:1:10;
fps        = 10;

%% 网格尺寸（示例）
nx_s=5; ny_s=5; nz_s=5;            % solid 网格点数（两块凝胶各自都是这个尺寸）
nx_f=101; ny_f=101; nz_f=101;      % fluid 网格点数

%% 网格步长与偏移（关键）
h_s = 1.0;
h_f = 0.5;
scale_f2s = h_f / h_s;
use_cell_center_vel = true;
offset_f = (use_cell_center_vel) * (0.5 * scale_f2s);

%% 显示/样式
c_lim_solid   = [0 0.2];
face_alpha    = 1.0;
quiver_step   = 5;
quiver_scale  = 1.0;
view_az_el    = [45 25];
do_hole_inside = true;

%% 工具：按 x快→y→z慢 向量 → (ny,nx,nz)
toYXZ = @(v,nx,ny,nz) permute(reshape(v,[nx,ny,nz]),[2 1 3]);

%% 视频
try
    vw = VideoWriter('gel_quiver_only_2gels.mp4','MPEG-4');
catch
    vw = VideoWriter('gel_quiver_only_2gels.avi','Motion JPEG AVI');
end
vw.FrameRate = fps; open(vw);

%% 图窗
fig = figure('Color','w','Units','pixels','Position',[60 40 600 550]);
ax  = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
set(gcf,'Renderer','opengl');

%% 预构造：流体箭头的“固体/物理坐标”网格
xF = (0:nx_f-1) * scale_f2s + offset_f;
yF = (0:ny_f-1) * scale_f2s + offset_f;
zF = (0:nz_f-1) * scale_f2s + offset_f;
[YF, XF, ZF] = ndgrid(yF, xF, zF);

% 稀疏取样索引
ix = 1:quiver_step:nx_f;
iy = 1:quiver_step:ny_f;
iz = 1:quiver_step:nz_f;
[IY, IX, IZ] = ndgrid(iy, ix, iz);
lin_samp = sub2ind([ny_f,nx_f,nz_f], IY, IX, IZ);
Xq = XF(lin_samp);  Yq = YF(lin_samp);  Zq = ZF(lin_samp);  % 箭头位置（固体坐标）

for t = frames
    % ---------- 文件名 ----------
    file_rm1 = fullfile(data_dir, sprintf(fmt_rm1, t));
    file_vm1 = fullfile(data_dir, sprintf(fmt_vm1, t));
    file_rm2 = fullfile(data_dir, sprintf(fmt_rm2, t));
    file_vm2 = fullfile(data_dir, sprintf(fmt_vm2, t));
    file_vb  = fullfile(data_dir, sprintf(fmt_velb, t));

    if ~(exist(file_rm1,'file') && exist(file_vm1,'file') && ...
         exist(file_rm2,'file') && exist(file_vm2,'file') && ...
         exist(file_vb ,'file'))
        fprintf('[skip] 缺少帧 %d 的 rm1/vm1 或 rm2/vm2 或 Velb\n', t);
        continue;
    end

    % ==========================================================
    %                 读入 凝胶 1
    % ==========================================================
    RM1 = load(file_rm1);    % [x y z]
    VM1 = load(file_vm1);    % [c]
    if size(RM1,2) < 3 || numel(VM1) ~= nx_s*ny_s*nz_s || size(RM1,1) ~= nx_s*ny_s*nz_s
        warning('帧 %d：rm1/vm1 尺寸不符，跳过', t); continue;
    end
    X1 = reshape(RM1(:,1), [nx_s,ny_s,nz_s]);
    Y1 = reshape(RM1(:,2), [nx_s,ny_s,nz_s]);
    Z1 = reshape(RM1(:,3), [nx_s,ny_s,nz_s]);
    C1 = reshape(VM1(:,1), [nx_s,ny_s,nz_s]);

    % 外表面六面（凝胶1）
    % z面
    X1z1 = X1(:,:,1);   Y1z1 = Y1(:,:,1);   Z1z1 = Z1(:,:,1);   C1z1 = C1(:,:,1);
    X1z2 = X1(:,:,end); Y1z2 = Y1(:,:,end); Z1z2 = Z1(:,:,end); C1z2 = C1(:,:,end);
    % y面
    X1y1 = squeeze(X1(:,1,:));  Y1y1 = squeeze(Y1(:,1,:));  Z1y1 = squeeze(Z1(:,1,:));  C1y1 = squeeze(C1(:,1,:));
    X1y2 = squeeze(X1(:,end,:));Y1y2 = squeeze(Y1(:,end,:));Z1y2 = squeeze(Z1(:,end,:));C1y2 = squeeze(C1(:,end,:));
    % x面
    X1x1 = squeeze(X1(1,:,:));  Y1x1 = squeeze(Y1(1,:,:));  Z1x1 = squeeze(Z1(1,:,:));  C1x1 = squeeze(C1(1,:,:));
    X1x2 = squeeze(X1(end,:,:));Y1x2 = squeeze(Y1(end,:,:));Z1x2 = squeeze(Z1(end,:,:));C1x2 = squeeze(C1(end,:,:));

    % ==========================================================
    %                 读入 凝胶 2
    % ==========================================================
    RM2 = load(file_rm2);    % [x y z]
    VM2 = load(file_vm2);    % [c]
    if size(RM2,2) < 3 || numel(VM2) ~= nx_s*ny_s*nz_s || size(RM2,1) ~= nx_s*ny_s*nz_s
        warning('帧 %d：rm2/vm2 尺寸不符，跳过', t); continue;
    end
    X2 = reshape(RM2(:,1), [nx_s,ny_s,nz_s]);
    Y2 = reshape(RM2(:,2), [nx_s,ny_s,nz_s]);
    Z2 = reshape(RM2(:,3), [nx_s,ny_s,nz_s]);
    C2 = reshape(VM2(:,1), [nx_s,ny_s,nz_s]);

    % 外表面六面（凝胶2）
    % z面
    X2z1 = X2(:,:,1);   Y2z1 = Y2(:,:,1);   Z2z1 = Z2(:,:,1);   C2z1 = C2(:,:,1);
    X2z2 = X2(:,:,end); Y2z2 = Y2(:,:,end); Z2z2 = Z2(:,:,end); C2z2 = C2(:,:,end);
    % y面
    X2y1 = squeeze(X2(:,1,:));  Y2y1 = squeeze(Y2(:,1,:));  Z2y1 = squeeze(Z2(:,1,:));  C2y1 = squeeze(C2(:,1,:));
    X2y2 = squeeze(X2(:,end,:));Y2y2 = squeeze(Y2(:,end,:));Z2y2 = squeeze(Z2(:,end,:));C2y2 = squeeze(C2(:,end,:));
    % x面
    X2x1 = squeeze(X2(1,:,:));  Y2x1 = squeeze(Y2(1,:,:));  Z2x1 = squeeze(Z2(1,:,:));  C2x1 = squeeze(C2(1,:,:));
    X2x2 = squeeze(X2(end,:,:));Y2x2 = squeeze(Y2(end,:,:));Z2x2 = squeeze(Z2(end,:,:));C2x2 = squeeze(C2(end,:,:));

    % ==========================================================
    %                 读入 fluid
    % ==========================================================
    VB = load(file_vb);        % [u v w]
    if size(VB,2) < 3 || size(VB,1) ~= nx_f*ny_f*nz_f
        warning('帧 %d：Velb 尺寸不符，跳过', t); continue;
    end
    Uf = toYXZ(VB(:,1), nx_f, ny_f, nz_f);
    Vf = toYXZ(VB(:,2), nx_f, ny_f, nz_f);
    Wf = toYXZ(VB(:,3), nx_f, ny_f, nz_f);

    % 取样到箭头位置
    Uq = Uf(lin_samp);  Vq = Vf(lin_samp);  Wq = Wf(lin_samp);

    % ---------- “挖空”：对两块凝胶的并集挖掉内部箭头 ----------
    if do_hole_inside
        try
            shp1 = alphaShape(X1(:), Y1(:), Z1(:));
            inside1 = inShape(shp1, Xq, Yq, Zq);

            shp2 = alphaShape(X2(:), Y2(:), Z2(:));
            inside2 = inShape(shp2, Xq, Yq, Zq);

            inside = inside1 | inside2;
            Uq(inside) = NaN; Vq(inside) = NaN; Wq(inside) = NaN;
        catch
            warning('alphaShape/inShape 不可用：未对凝胶内部挖空（帧 %d）', t);
        end
    end

    % ==========================================================
    %                       绘制
    % ==========================================================
    cla(ax); hold(ax,'on');

    % (A) 凝胶1 六面
    surf(ax, X1z1, Y1z1, Z1z1, C1z1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X1z2, Y1z2, Z1z2, C1z2, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X1y1, Y1y1, Z1y1, C1y1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X1y2, Y1y2, Z1y2, C1y2, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X1x1, Y1x1, Z1x1, C1x1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X1x2, Y1x2, Z1x2, C1x2, 'EdgeColor','none','FaceAlpha',face_alpha);

    % (B) 凝胶2 六面
    surf(ax, X2z1, Y2z1, Z2z1, C2z1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X2z2, Y2z2, Z2z2, C2z2, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X2y1, Y2y1, Z2y1, C2y1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X2y2, Y2y2, Z2y2, C2y2, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X2x1, Y2x1, Z2x1, C2x1, 'EdgeColor','none','FaceAlpha',face_alpha);
    surf(ax, X2x2, Y2x2, Z2x2, C2x2, 'EdgeColor','none','FaceAlpha',face_alpha);

    colormap(ax, 'jet'); caxis(ax, c_lim_solid); colorbar(ax);

    % (C) 外部流场箭头
    quiver3(ax, Xq, Yq, Zq, Uq, Vq, Wq, ...
        'AutoScale','on','AutoScaleFactor',quiver_scale, 'LineWidth',0.9);

    % 轴范围：考虑两块凝胶+流体
    xmin = min([min(X1(:)) min(X2(:)) min(XF(:))]);
    xmax = max([max(X1(:)) max(X2(:)) max(XF(:))]);
    ymin = min([min(Y1(:)) min(Y2(:)) min(YF(:))]);
    ymax = max([max(Y1(:)) max(Y2(:)) max(YF(:))]);
    zmin = min([min(Z1(:)) min(Z2(:)) min(ZF(:))]);
    zmax = max([max(Z1(:)) max(Z2(:)) max(ZF(:))]);
    pad = 0.02 * max([xmax-xmin, ymax-ymin, zmax-zmin]);
    axis(ax, [xmin-pad xmax+pad ymin-pad ymax+pad zmin-pad zmax+pad]);
    daspect(ax, [1 1 1]);
    set(gca, 'FontSize', 20);
    %view(ax, view_az_el(1), view_az_el(2));
    view(0,90);
    xlabel(ax,'X'); ylabel(ax,'Y'); zlabel(ax,'Z');
    title(ax, sprintf('Time =  %d', t));

    drawnow;
    writeVideo(vw, getframe(fig));
end

close(vw);
fprintf('gel_quiver_only_2gels.mp4\n');

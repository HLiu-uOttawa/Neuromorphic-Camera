classdef EventH5Reader
    properties
        file
        t0_ns
    end

    methods
        function obj = EventH5Reader(h5file)
            obj.file = h5file;
            obj.t0_ns = h5readatt(h5file, '/', 't0_ns');
        end

        function s = summary(obj)
            s = struct();
            s.file = obj.file;
            s.t0_ns = obj.t0_ns;
            info = h5info(obj.file);
            s.groups = {info.Groups.Name};
        end

        function ev = readEventsRel(obj, t0_rel_ns, t1_rel_ns, maxN)
            % Read events in [t0_rel_ns, t1_rel_ns]
            if nargin < 4, maxN = inf; end

            t = h5read(obj.file, '/events/t_ns_rel');
            idx = find(t >= t0_rel_ns & t <= t1_rel_ns);

            if isempty(idx)
                ev = struct('t', [], 'x', [], 'y', [], 'p', []);
                return;
            end

            if ~isinf(maxN) && numel(idx) > maxN
                idx = idx(1:maxN);
            end

            ev.t = t(idx);
            ev.x = h5read(obj.file, '/events/x', idx(1), numel(idx));
            ev.y = h5read(obj.file, '/events/y', idx(1), numel(idx));
            ev.p = h5read(obj.file, '/events/p', idx(1), numel(idx));
        end

        function fr = readFrameByIndex(obj, k)
            fr.t = h5read(obj.file, '/frames/t_ns_rel', k, 1);

            ds = h5info(obj.file, '/frames/image');
            sz = double(ds.Dataspace.Size);

            % Case A: stored as (C,H,W,K) like [3 346 260 3505]
            if numel(sz) == 4 && sz(1) == 3
                K = sz(4);
                if k < 1 || k > K
                    error("Frame index k=%d out of range [1,%d].", k, K);
                end

                start = [1 1 1 double(k)];
                count = [3 sz(2) sz(3) 1];

                img = h5read(obj.file, '/frames/image', start, count); % (3,H,W,1)
                img = squeeze(img);                                   % (3,H,W)
                img = permute(img, [2 3 1]);                           % (H,W,3)
                fr.image = img;
                return;
            end

            % Case B: stored as (K,H,W[,C]) (future-proof)
            nd = numel(sz);
            if k < 1 || k > sz(1)
                error("Frame index k=%d out of range [1,%d].", k, sz(1));
            end
            start = ones(1, nd, 'double'); start(1) = double(k);
            count = sz; count(1) = 1;
            img = h5read(obj.file, '/frames/image', start, count);
            fr.image = squeeze(img);
        end



        function imu = readImuRel(obj, t0_rel_ns, t1_rel_ns)
            t = h5read(obj.file, '/imu/t_ns_rel');
            idx = find(t >= t0_rel_ns & t <= t1_rel_ns);

            if isempty(idx)
                imu = struct('t', [], 'accel', [], 'gyro', []);
                return;
            end

            imu.t = t(idx);

            szA = double(h5info(obj.file, '/imu/accel').Dataspace.Size);
            szG = double(h5info(obj.file, '/imu/gyro').Dataspace.Size);

            i0 = double(idx(1));
            n  = double(numel(idx));

            % accel
            if numel(szA)==2 && szA(1)==3
                % stored as (3,M)
                A = h5read(obj.file, '/imu/accel', [1 i0], [3 n]);  % 3×n
                imu.accel = A.';                                    % n×3
            else
                % stored as (M,3)
                imu.accel = h5read(obj.file, '/imu/accel', [i0 1], [n 3]);
            end

            % gyro
            if numel(szG)==2 && szG(1)==3
                G = h5read(obj.file, '/imu/gyro', [1 i0], [3 n]);   % 3×n
                imu.gyro = G.';                                     % n×3
            else
                imu.gyro = h5read(obj.file, '/imu/gyro', [i0 1], [n 3]);
            end
        end


        function tr = readTriggersRel(obj, t0_rel_ns, t1_rel_ns)
            t = h5read(obj.file, '/triggers/t_ns_rel');
            idx = find(t >= t0_rel_ns & t <= t1_rel_ns);

            if isempty(idx)
                tr = struct('t', [], 'id', [], 'value', []);
                return;
            end

            tr.t = t(idx);
            tr.id = h5read(obj.file, '/triggers/id', idx(1), numel(idx));
            tr.value = h5read(obj.file, '/triggers/value', idx(1), numel(idx));
        end
    end

    % ==============================================================
    % Static self-test (RadarDataWindow-style)
    % ==============================================================
    methods (Static)
        function selfTest(h5file)
            % Static self-test for EventH5Reader
            % Usage:
            %   EventH5Reader.selfTest("file.h5")
            %   EventH5Reader.selfTest()   % popup dialog

            if nargin < 1 || isempty(h5file)
                [fname, fpath] = uigetfile( ...
                    {'*.h5;*.hdf5', 'HDF5 files (*.h5, *.hdf5)'}, ...
                    'Select Event HDF5 file');
                if isequal(fname, 0)
                    fprintf("SelfTest cancelled by user.\n");
                    return;
                end
                h5file = fullfile(fpath, fname);
            end

            fprintf("=== EventH5Reader SELF TEST ===\n");
            fprintf("File: %s\n", h5file);

            r = EventH5Reader(h5file);

            % ---- summary ----
            s = r.summary();
            disp(s);

            % ---- frames ----
            if any(strcmp(s.groups, '/frames'))
                fprintf("[OK] frames group found\n");
                fr = r.readFrameByIndex(1);
                fprintf("  frame t (rel ns): %d\n", fr.t);

                figure('Name', 'Event Frame');
                % imshow(uint8(fr.image));
                img = fr.image;

                % normalize type for display
                if ~isa(img, 'uint8')
                    % 如果是 uint16 / int16 / float，先转换到 double 再归一化
                    imgd = double(img);
                    imgd = imgd - min(imgd(:));
                    if max(imgd(:)) > 0
                        imgd = imgd ./ max(imgd(:));
                    end
                else
                    imgd = img;
                end

                nd = ndims(img);
                sz = size(img);

                fprintf("  frame image size: %s\n", mat2str(sz));

                figure('Name', 'Event Frame');

                if nd == 2
                    imshow(imgd); title("First frame (grayscale)");
                elseif nd == 3
                    C = sz(3);
                    if C == 3
                        imshow(imgd); title("First frame (RGB)");
                    else
                        % 多平面但不是RGB：默认显示第1通道（最稳）
                        imshow(imgd(:,:,1));
                        title(sprintf("First frame (channel 1 of %d)", C));
                    end
                else
                    % 更高维：先 squeeze，再按前两维显示
                    img2 = squeeze(imgd);
                    if ndims(img2) >= 2
                        imshow(img2(:,:,1));
                        title("First frame (squeezed, channel 1)");
                    else
                        warning("Frame image has unexpected shape; cannot display.");
                    end
                end

                title("First event-camera frame");
            else
                fprintf("[INFO] no frames group\n");
            end

            % ---- events ----
            if any(strcmp(s.groups, '/events'))
                fprintf("[OK] events group found\n");
                ev = r.readEventsRel(0, 5e8, 50000); % first 0.5 s, max 50k
                fprintf("  events read: %d\n", numel(ev.t));

                figure('Name', 'Event Scatter');
                scatter(ev.x, ev.y, 5, ev.p, 'filled');
                axis equal ij;
                xlabel('x'); ylabel('y');
                title('Event scatter (first 0.5 s)');
            else
                fprintf("[INFO] no events group\n");
            end

            % ---- imu ----
            if any(strcmp(s.groups, '/imu'))
                fprintf("[OK] imu group found\n");
                imu = r.readImuRel(0, 1e9);
                fprintf("  imu samples: %d\n", numel(imu.t));
            else
                fprintf("[INFO] no imu group\n");
            end

            % ---- triggers ----
            if any(strcmp(s.groups, '/triggers'))
                fprintf("[OK] triggers group found\n");
                tr = r.readTriggersRel(0, 1e9);
                fprintf("  triggers: %d\n", numel(tr.t));
            else
                fprintf("[INFO] no triggers group\n");
            end

            fprintf("=== SELF TEST PASSED ===\n");
        end
    end

end

% EventH5Reader.selfTest("ottawa_8.h5");
% or:
% EventH5Reader.selfTest();
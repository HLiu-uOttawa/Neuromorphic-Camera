% read_image_pixels.m
% Read an image and get pixel color values (RGB)

clear; clc; close all;

% 1. Read image
img = imread('7_Video_7_frame_31566351.png');   % <-- 替换为你的图片路径

% 2. Show image
figure;
imshow(img);
title('Input Image');

% 3. Get image size
[H, W, C] = size(img);

if C ~= 3
    error('This example assumes an RGB image.');
end

% 4. Preallocate array for pixel values
% pixel_values(y, x, :) = [R G B]
pixel_values = zeros(H, W, 3, 'uint8');

% 5. Loop through pixels
for y = 1:H
    for x = 1:W
        R = img(y, x, 1);
        G = img(y, x, 2);
        B = img(y, x, 3);

        pixel_values(y, x, :) = [R, G, B];

        % Uncomment if you want to print every pixel
        fprintf('Pixel (%d, %d): R=%d G=%d B=%d\n', x, y, R, G, B);
    end
end

disp('Finished reading all pixel values.');

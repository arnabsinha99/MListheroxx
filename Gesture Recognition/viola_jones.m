clc;
clear all;
close all;
a = imread('Images/frame1_1.jpg');
b = vision.CascadeObjectDetector();
bbox = step(b,a);
imshow(a);
rectangle('position',bbox);
b11 = imcrop(a);
b12 = rgb2hsv(b11);
h = b12(:,:,1);
havg = mean(mean(h)); 
converted = rgb2hsv(a);
leftext = havg - (0.2*havg);
rightext = havg + (0.2*havg);
hfinal = converted(:,:,1);

[w,h] = size(a(:,:,1));
new_image = zeros(size(a(:,:,1)));

for i = 1:w
    for j = 1:h        
        pixel = hfinal(i,j);
        if(pixel < leftext) || (pixel > rightext)
            new_image(i,j) = 0;
        else
            new_image(i,j) = 1;
        end
    end
end

imshow(new_image);

a = double(a);
a(:,:,1) = a(:,:,1).*new_image;
a(:,:,2) = a(:,:,1).*new_image;
a(:,:,2) = a(:,:,1).*new_image;

a = uint8(a);

imshow(a);

function [features] = ExtractSIFTfeatures(image, x, y, feature_width)
f = feature_width/2;
f1 = feature_width/4;
s = size(x);
si = size(image); 
image_rows = si(1);
image_cols = si(2);
features = zeros(s(1),128);

%Rotating the Sobel Filter 
filter = [];
m = fspecial('sobel'); 
filter(:,:,1) = m;
for i=2:8 
    m = [m(4) m(7) m(8); m(1) m(5) m(9); m(2) m(3) m(6)];
    filter(:,:,i) = m;
end

%Constructing New Image
im_size = size(image);
new_image = zeros(im_size(1),im_size(2),8); 
for i=1:8 
    new_image(:,:,i) = imfilter(image,filter(:,:,i)); 
end
blur = fspecial('gaussian', [feature_width/2, feature_width/2], feature_width / 2);
new_image = imfilter(new_image,blur);

for i=1:s(1)
    x_coord = uint16(x(i));
    y_coord = uint16(y(i));     
    x_left = x_coord-f;
    x_right = x_coord+f-1;
    y_top = y_coord-f;
    y_bottom= y_coord+f-1;
    bins = zeros(1,128);
    if x_left >= 1 & x_right <= image_cols-f & y_top >=1 & y_bottom <= image_rows-f  
        for j=1:8
        	image = new_image(:,:,j); 
            patch = image(y_top:y_bottom, x_left:x_right);
            small_square = mat2cell(patch,[f1,f1,f1,f1], [f1,f1,f1,f1]);
            c = j;
            for row=1:4 
                for col=1:4
                    cell = cell2mat(small_square(row,col));
                    val = sum(cell(:));
                    bins(:,c) = val;
                    c = c+8;
                end
            end
            end 
        bins = bins./norm(bins);
        features(i,:) = bins(1,:); 
        end
    end 
end
figure(1)
title('Patch plots');
for i=1:10
subplot(2,5,i);
imshow(patch(:,:,i),[])
    
end


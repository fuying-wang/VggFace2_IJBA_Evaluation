clear;clc;close all;
cd('../')

%% caffe setttings
matCaffe = fullfile(pwd, '../tools/caffe-sphereface/matlab');
addpath(genpath(matCaffe));
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();

model   = '../train/code/sphereface_deploy.prototxt';
weights = '../train/result/sphereface_model_iter_28000.caffemodel';
net     = caffe.Net(model, weights, 'test');
net.save('result/sphereface_model.caffemodel');


function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    res_    = net.forward({flip(img, 1)});
    feature = double([res{1}; res_{1}]);
end

% Iteratively extracts the features of each image and save them

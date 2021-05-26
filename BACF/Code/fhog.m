function H = fhog( I, binSize, nOrients, clip, crop )
% Efficiently compute Felzenszwalb's HOG (FHOG) features.
%
% A fast implementation of the HOG variant used by Felzenszwalb et al.
% in their work on discriminatively trained deformable part models.
%  http://www.cs.berkeley.edu/~rbg/latent/index.html
% Gives nearly identical results to features.cc in code release version 5
% but runs 4x faster (over 125 fps on VGA color images).
%
% The computed HOG features are 3*nOrients+5 dimensional. There are
% 2*nOrients contrast sensitive orientation channels, nOrients contrast
% insensitive orientation channels, 4 texture channels and 1 all zeros
% channel (used as a 'truncation' feature). Using the standard value of
% nOrients=9 gives a 32 dimensional feature vector at each cell. This
% variant of HOG, refered to as FHOG, has been shown to achieve superior
% performance to the original HOG features. For details please refer to
% work by Felzenszwalb et al. (see link above).
%
% This function is essentially a wrapper for calls to gradientMag()
% and gradientHist(). Specifically, it is equivalent to the following:
%  [M,O] = gradientMag( I,0,0,0,1 ); softBin = -1; useHog = 2;
%  H = gradientHist(M,O,binSize,nOrients,softBin,useHog,clip);
% See gradientHist() for more general usage.
%
% This code requires SSE2 to compile and run (most modern Intel and AMD
% processors support SSE2). Please see: http://en.wikipedia.org/wiki/SSE2.
%
% USAGE
%  H = fhog( I, [binSize], [nOrients], [clip], [crop] )
%
% INPUTS
%  I        - [hxw] color or grayscale input image (must have type single)
%  binSize  - [8] spatial bin size
%  nOrients - [9] number of orientation bins
%  clip     - [.2] value at which to clip histogram bins
%  crop     - [0] if true crop boundaries
%
% OUTPUTS
%  H        - [h/binSize w/binSize nOrients*3+5] computed hog features
%
% EXAMPLE
%  I=imResample(single(imread('peppers.png'))/255,[480 640]);
%  tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc) % >125 fps
%  figure(1); im(I); V=hogDraw(H,25,1); figure(2); im(V)
%
% EXAMPLE
%  % comparison to features.cc (requires DPM code release version 5)
%  I=imResample(single(imread('peppers.png'))/255,[480 640]); Id=double(I);
%  tic, for i=1:100, H1=features(Id,8); end; disp(100/toc)
%  tic, for i=1:100, H2=fhog(I,8,9,.2,1); end; disp(100/toc)
%  figure(1); montage2(H1); figure(2); montage2(H2);
%  D=abs(H1-H2); mean(D(:))
%
% See also hog, hogDraw, gradientHist
%
% Piotr's Image&Video Toolbox      Version 3.23
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

%Note: modified to be more self-contained

if( nargin<2 ), binSize=8; end
if( nargin<3 ), nOrients=9; end
if( nargin<4 ), clip=.2; end
if( nargin<5 ), crop=0; end

softBin = -1; useHog = 2; b = binSize;

[M,O]=gradientMex('gradientMag',I,0,1);

H = gradientMex('gradientHist',M,O,binSize,nOrients,softBin,useHog,clip);

if( crop ), e=mod(size(I),b)<b/2; H=H(2:end-e(1),2:end-e(2),:); end

end
%一下是pca降维过程
function [newX,T,meanValue] = pca_row(X,CRate)
%每行是一个样本
%newX  降维后的新矩阵
%T 变换矩阵
%meanValue  X每列均值构成的矩阵，用于将降维后的矩阵newX恢复成X
%CRate 贡献率
%计算中心化样本矩阵
meanValue=ones(size(X,1),1)*mean(X);
X=X-meanValue;%每个维度减去该维度的均值
C=cov(x);%计算协方差矩阵
 
%计算特征向量V，特征值D
[V,D]=eig(C);
%将特征向量按降序排序
[dummy,order]=sort(diag(D),'descend');
V=V(:,order);%将特征向量按照特征值大小进行降序排列
d=diag(D);%将特征值取出，构成一个列向量
newd=d(order);%将特征值构成的列向量按降序排列
 
%取前n个特征向量，构成变换矩阵
sumd=sum(newd);%特征值之和
for j=1:length(newd)
    i=sum(newd(1:j,1))/sumd;%计算贡献率，贡献率=前n个特征值之和/总特征值之和
    if i>CRate%当贡献率大于95%时循环结束,并记下取多少个特征值
        cols=j;
        break;
   end
end
T=V(:,1:cols);%取前cols个特征向量，构成变换矩阵T
newX=X*T;%用变换矩阵T对X进行降维
end


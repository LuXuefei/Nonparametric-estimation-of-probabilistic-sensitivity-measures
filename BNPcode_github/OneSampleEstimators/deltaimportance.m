function d=deltaimportance(x,y,M)
% DELTAIMPORTANCE a quick method of computing delta moment independent measure
% DELTA=DELTAIPORTANCE(X,Y)
%%
[n,k]=size(x);
% Epanechnikov with variance 1
Kernel=@(x)3/(4*sqrt(5))*max(1-(x.^2/5),0);
% Box kernel
%Kernel=@(x)(1-abs(x/sqrt(3))>0)/(2*sqrt(3));

% Numerical noise cutoff (simple Kolmogorov-Smirnov)
Cutoff = 1; 

%% transform to Gaussian
[ty,iy]=sort(y);yr(iy)=1:n;
ty(iy)=-sqrt(2)*erfinv(1-(2*(1:n)-1)'/n);
%% work with tranformed data
medy=median(ty);
iqry=median(abs(medy-ty)); % interquartile range estimator
stdy=min(std(ty),iqry/0.675);
% bandwidth estimate (optimal for Gaussian data/Gaussian kernel)
h=stdy/(((3*n)/4)^(1/5));
% constuct interpolation points
z1=linspace(min(ty)-h,medy-iqry, 25);
z2=linspace(medy-iqry,medy+iqry,52);
z3=linspace(medy+iqry,max(ty)+h,25);
z=[z1,z2(2:end-1),z3];
l=length(z);
%% kernel density matrix
W=Kernel( (repmat(z,n,1)-repmat(ty,1,l))/h)/h;
%% unconditional density 
densy=mean(W);
%% conditional densities for partitioned data
%M=50;
m=linspace(0,n,M+1);
Sm=zeros(k,M);
nm=zeros(k,M);

%% select W entries from the partition
[xr,indxx]=sort(x);
for i=1:k
   xr(indxx(:,i),i)=1:n; % ranks (no ties)
end
%%
for j=1:M
   indx= (m(j)<xr) & (xr <= m(j+1));
   nm(:,j)=sum(indx); % with no ties: always same nr. of realizations
   for i=1:k
   % conditional density
   densc=mean(W(indx(:,i),:));
   % L1 separation of densities (using Scheffe Thm)
   Sm(i,j)=trapz(z,max(densy-densc,0)); % all entries <1
   end
end
% Clear noise
 Sm(Sm<Cutoff.*sqrt(1/n+1./nm))=0;
 d=sum(Sm.*nm,2)'/n;
end

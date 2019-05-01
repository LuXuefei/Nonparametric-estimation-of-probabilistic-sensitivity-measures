function [delta,eta,ce,ks,D,E,KL,KS]=OneSampleBBPolyaUrn(x,y,h,RS)
%% Partition-dependent Polya Urn estimator
%% for {Borgonovo delta, first order Sobol, Cross entropy(KullbackĘCLeibler), Kolmogorov-Smirnov}
%% *Inputs:
% x: model inputs (n * k) ;
% y: scalar model output n*1 ;
% h: number of partition sets ;
% RS: posterior sample size;
%% *Outputs:
% delta, eta, ce, ks: point estimates
% D,E,KL,KS: posterior sample for correponding sensitivity measures
%%
%parpool % allow parallel computing
rng(12345); % comment out if unnecessary
[n,K]=size(x);
if(nargin < 3),h = ceil(n^(1/3));end
if(nargin < 4),RS = 100;end
SD = zeros(h,1);  SE = zeros(h,1);  SKL = zeros(h,1);  SKS = zeros(h,1);
D = zeros(RS,K);  KL = zeros(RS,K);  KS = zeros(RS,K);  E = zeros(RS,K);
ym = mean(y); yd = var(y);
%%
for w = 1:RS
    for k = 1:K % nr of input
        xx = x(1:n,k); yy = y(1:n,1);
        M = [xx yy];
        MM = sortrows(M); %sort by xx
        W = floor(n/h);
        % Domain vector of Y for trapezoidal numerical integration
        yq = linspace(min(yy),max(yy),300);
        % Unconditional kernel-smooth pdf $f_{Y}$
        f_Yev = ksdensity(yy,yq);
        for m=1:h
            YX=MM((m-1)*W+1:min(m*W,n),2); % $\mathbb{y}$ given partition X_m
            if(m==h) YX=MM((m-1)*W+1:n,2); end
            nm = size(YX,1);
            alpha = 0.1*nm; % Concentration parameter
            for i=  1: n-nm
                P1 = alpha/(alpha+nm+i-1);
                u = rand(1);
                if u <= P1
                    YX(nm+i) = normrnd(mean(YX),std(YX)); % Base measure G
                else
                    YX(nm+i) = datasample(YX,1); % P_n as empirical YX
                end
            end
            %delta
            f_YXev =ksdensity(YX,yq); ffd= abs(f_Yev-f_YXev);
            SD(m) =  trapz(yq,ffd);
            %eta
            SE(m) = nm*(mean(YX)-ym)^2;
            %CE/ KL
            ffKL =(log(f_YXev)-log(f_Yev)).*f_YXev;ffKL(f_YXev==0)=0;
            SKL(m)=trapz(yq,ffKL);
            %KS
            [H,P,SKS(m)]=kstest2(yy,YX);
        end
        D(w,k) = mean(SD)/2;
        KL(w,k)=mean(SKL);
        KS(w,k)=mean(SKS);
        E(w,k) = sum(SE)/yd/(n-1);
    end
end
delta = mean(D,1);
ce= mean(KL,1);
eta = mean(E,1);
ks = mean(KS,1);
end % end of function

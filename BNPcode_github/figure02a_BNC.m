%% Figure 02(a) %% 
%%
% * Partition-free Bayesian non-parametric conditional estimates the 21-input Additive Gassian simulator:
%                            $Y=\sum_{i=1}^{21}a_i X_i $
% where X_i ~ Normal(1; 1), with a1 = ... = a7 = -4, a8 =... = a14 = 2, and 
% a15 = ... = a21 = 1. The 21 inputs are correlated with \rho(i;j) = 0.5
%
% * Reference: 
%  Isadora Antoiano-Villalobos, Emanuele Borgonovo, Xuefei Lu(2019). 
%  Nonparametric estimation of probabilistic sensitivity measures.
%
% * Author: Xuefei Lu, xuefei.lu@unibocconi.it
% * Date: May, 2019
%%
% Clear workspace
clearvars
addpath('./cond/')
% The folder for functions NPRegNW and NPRegNWfF, which are algorithms from
% the work of 
% Antoniano-Villalobos I, Wade S, Walker SG. A Bayesian nonparametric regression model with normalized weights: a study of hippocampal atrophy in Alzheimer’s disease. 
% Journal of the American Statistical Association. 2014 Apr 3;109(506):477-90.
% Those codes are available from the authors.
addpath('./BNCestimator/')
addpath('./datasets')

% Set random number generator seed
seednr = 13579;


% Load data
x = load('x_AGcor_5000.txt'); %input
y = load('y_AGcor_5000.txt'); %output

% % Sample size
n=600;
% % Choose the i-th input realizations
kk = 3; %[3,10,18] input index
x=x(1:n,kk);
% Output realizations
Y=y(1:n);

%% Domain vector of Y for trapezoidal numerical integration
n_new = 400;
x_new =linspace(min(x)-0.01*range(x), max(x)+0.01*range(x), n_new);
x_new = x_new';

n_newy = 500;
Y_grid = linspace(min(y)-0.01*range(y), max(y)+0.01*range(y), n_newy);
Y_grid = Y_grid';
ny=length(Y_grid);

%% Initialize Prior Parameters
% Number of discrete covariates
q=0;
% Number of continuous covariates
p=1;

% MCMC posterior sample size
S=2000; %
% Burn in period
burnin=3000;
% Thinning
thin=1;
mcmc=[burnin,thin,S];
clear burnin thin

% Base measure hyperparameters
[beta0, iC] = choosehyperV2(x,Y);
iC = 1*iC;

% alpha1/alpha2 is the mean for 1/sigma^2
alpha1=1; alpha2=1;

% Hyperparameters for the Normal-Gamma prior for the (mu_j,tau)
mu0=1; %mean of x
c=1;
a1=1; a2=1; % a1/a2 is the mean for tau

% Generate data structure for NPRegNW function and clear auxiliary
% variables
Hyperparameters=cell(5,1);
Hyperparameters{1}=beta0;
Hyperparameters{2}=iC;
Hyperparameters{3}=[alpha1,alpha2];
Hyperparameters{4}=[mu0,c,a1,a2];

% OPTIONAL PARAMETERS %%
% Slice sampling constant
phi=1;
% Nonparametric prior specification
PP=cell(2,1);
PP{1}='StickBreaking';
PP{2}=[1,1];
% Initial state: randomized alternative
cl=3; % Number of random initial clusters
da=discreternd((1:cl)',n)'; % only one component with every observation associated to it

% Latent model variables
ka=5*ones(n,1); % No latent variables
Da=cell(n,1); 
for i=1:n
    Da{i}=discreternd((1:2*cl)',ka(i))';
end

% Parameters for continuous covariates
taua=a1./a2; % precision set at prior mean
rng(1234)
mua=randn(1,2*cl)./sqrt(taua*c)+mu0;

% Generate data structure for NPRegNW function and clear auxiliary
% variables
initialState=cell(5,1);
initialState{1}=da;
initialState{2}=ka;
initialState{3}=Da;
initialState{4}=mua;
initialState{5}=taua;
if strcmp(PP{1},'DirichletHP')
    initialState{6}=massa;
    clear massa
end
%% Posterior sampling
rng(seednr)
[w,theta,J,d,cputime,lastState,PP,phi,initialState]=NPRegNW(Y,x,q,p,mcmc,Hyperparameters,'Initial',initialState, 'Prior',PP,'Phi',phi);
% Functions NPRegNW and NPRegNWfF are codes from
% Antoniano-Villalobos I, Wade S, Walker SG. A Bayesian nonparametric regression model with normalized weights: a study of hippocampal atrophy in Alzheimer’s disease. 
% Journal of the American Statistical Association. 2014 Apr 3;109(506):477-90.
%% BNP conditional density estimates
rng(seednr)
[Y_fpred,fpred_Yxnew,Y_fpredF,fpred_YxnewF]=NPRegNWfF(Y_grid,x_new,q,p,w,theta,J);
%% Prepare required distributions

% True input distribution, needs to be changed accroding to the simulator
fx = normpdf(x_new,1,1);

% extract posterior sample of $f_{Y|X}$ at each iteration
 fpred_YX = cell(S,1);  
 for s=1:S
     fYX0=nan(n_new,length(Y_grid));
    for k=1:n_new
       fYx = fpred_Yxnew{k};
       fYX0(k,:)=fYx(s,:);
    end    
    fpred_YX{s}=fYX0;
 end
 
% extract posterior sample of $F_{Y|X}$ at each iteration
 Fpred_YX = cell(S,1); 
 for s=1:S
     FYX0=nan(n_new,length(Y_grid));
    for k=1:n_new
       FYx = fpred_YxnewF{k};
       FYX0(k,:)=FYx(s,:);
    end    
    Fpred_YX{s}=FYX0;
 end

% Marginal pdf of Y
fY = cell(S,1);
 for s=1:S
    fYX0= fpred_YX{s}.*repmat(fx,1,ny);
    fY{s}=trapz(x_new,fYX0,1); 
 end 
 
% Marginal cdf of Y
  FY = cell(S,1);
 for s=1:S
    FYX0= Fpred_YX{s}.*repmat(fx,1,ny);
    FY{s}=trapz(x_new,FYX0,1); 
 end
 
%% $\delta$
delta = zeros(S,1); 
for s=1:S
    fint = abs( (repmat(fY{s},n_new,1)- fpred_YX{s}).*repmat(fx,1,ny) );
    delta(s) =0.5* trapz(x_new,trapz(Y_grid,fint,2));
end
% point estimate, \delta^{BNC}
mean(delta)
% 95% credibility interval
prctile(delta,[2.5 97.5])

%% $\beta^{KS}$
betaKS = zeros(S,1); 
for s=1:S
    fint=max(abs(repmat(FY{s},n_new,1)- Fpred_YX{s}),[],2);
    betaKS(s)=trapz(x_new,fint.*fx);
end
% point estimate, \beta^{BNC}
mean(betaKS)
% 95% credibility interval
prctile(betaKS,[2.5 97.5])

%% variance-based $\eta$
% \mu_{Y}^s(x):= E[Y|x, \theta]
rng(seednr)
[Y_pred,Y_predS]=NPRegNWprediction(x_new,q,p,w,theta,J);

% \tilde{\mu}_{Y}^s:= E[\mu_{Y}^s(X)]
EEYX=zeros(S,1); 
for s=1:S
   EEYX(s)= trapz(x_new,Y_predS(s,:).*fx');
end

% V^s:= V[\mu_{Y}(X)]
VEYX=zeros(S,1); 
for s=1:S
 VEYX(s) = trapz(x_new,(Y_predS(s,:)-EEYX(s)).^2.*fx');
end 

% \mu_{Y}^s := E[Y|\theta_{s}]
% V_{Y}^s:= V[Y|\theta_{s}]
EY=zeros(S,1); VY = zeros(S,1);
for s=1:S
   EY(s)=trapz(Y_grid, Y_grid'.*fY{s});
   VY(s) = trapz(Y_grid, (Y_grid-EY(s))'.^2.*fY{s});
end

eta = VEYX./VY;
% point estimate, \eta^{BNC}
mean(eta)
% 95% credibility interval
prctile(eta,[2.5 97.5])


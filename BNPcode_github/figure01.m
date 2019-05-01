%% Figure 01 %% 
%%
% * Partition-dependent global sensitivity measures estimates for the 21-input Additive Gassian simulator:
%                            $Y=\sum_{i=1}^{21}a_i X_i $
% where X_i ~ Normal(1; 1), with a1 = ... = a7 = -4, a8 =... = a14 = 2, and 
% a15 = ... = a21 = 1. The 21 inputs are correlated with \rho(i;j) = 0.5

% frequintist pdf/cdf-based estimators and 
% partition-dependent Bayesian bootstrap/ Polya urn estimators.
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
% Set random number generator seed
rng(123456); 
addpath('.\OneSampleEstimators\')

% Load data
addpath('.\datasets\')
x = load('x_AGcor_5000.txt');
x = x(:,[3,10,18]);
y = load('y_AGcor_5000.txt');


% Sample size
n=900; 
% Input dimension
k=size(x,2);



%% Frequintist pdf/cdf-based estimators

% Number of partition sets
M = 10; 

% Pdf-based estimates for variance-based $\eta$
[~,EstPrevETA1] = deltamim(x(1:n,:),y(1:n),M); 
% deltamin.m is the code from
% Borgonovo E, Castaings W, Tarantola S. Moment independent importance measures: New results and analytical test cases. 
% Risk Analysis. 2011 Mar 1;31(3):404-28.

% Pdf-based estimates for $\delta$-portance measure
EstPrevBD1 = deltaimportance(x(1:n,:),y(1:n),M); 
% deltaimportance.m is the code from 
% Plischke E, Borgonovo E, Smith CL. Global sensitivity measures from given data. 
% European Journal of Operational Research. 2013 May 1;226(3):536-50.

% Cdf-based estimates for $\beta^{KS}$,$\delta$-portance measure and
%variance-based $\eta$
[EstPrevKS2,EstPrevBD2,~,EstPrevETA2]=betaKS3(x(1:n,:),y(1:n),M);
% betaKS3.m is the code from
% Plischke E, and Borgonovo E. Probabilistic Sensitivity Measures from Empirical
% Cumulative Distribution Functions. Work in Progress. 2017.

%% Partition-dependent Bayesian bootstrap and Polya Urn estimators

% Bootstrap sample size
S = 100;

%% Bayesian bootstrap
[deltaBb,etaBb,ceBb,ksBb,DBb,EBb,KLBb,KSBb]=OneSampleBBPosterior(x(1:n,:),y(1:n),M,S);
% point estimates for $\eta$
etaBb
% 95% credibility interval
prctile(EBb,[2.5 97.5])

% point estimates for $\delta$
deltaBb
% 95% credibility interval
prctile(DBb,[2.5 97.5])

% point estimates for $\beta^{KS}$
ksBb
% 95% credibility interval
prctile(KSBb,[2.5 97.5])


%% Polya Urn
[deltaPu,etaPu,cePu,ksPu,DPu,EPu,KLPu,KSPu]=OneSampleBBPolyaUrn(x(1:n,:),y(1:n),M,S);
% point estimates for $\eta$
etaPu
% 95% credibility interval
prctile(EPu,[2.5 97.5])

% point estimates for $\delta$
deltaPu
% 95% credibility interval
prctile(DPu,[2.5 97.5])

% point estimates for $\beta^{KS}$
ksPu
% 95% credibility interval
prctile(KSPu,[2.5 97.5])




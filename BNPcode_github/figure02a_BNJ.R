## Figure 02(a) ##
##
# * Partition-free Bayesian non-parametric joint estimates for the 21-input simulator:
#                            $Y=\sum_{i=1}^{21}a_i X_i, X_i~iid Normal(1,1)$
# where X_i ~ Normal(1; 1), with a1 = ... = a7 = -4, a8 =... = a14 = 2, and 
# a15 = ... = a21 = 1. The 21 inputs are correlated with \rho(i;j) = 0.5
#
# * Reference:
#  Isadora Antoiano-Villalobos, Emanuele Borgonovo, Xuefei Lu(2019).
#  Nonparametric estimation of probabilistic sensitivity measures.
#
# * Author: Xuefei Lu, xuefei.lu@unibocconi.it
# * Date: May, 2019


library(DPpackage)
library(MASS)
library(plyr)
library(mvtnorm)
library(cubature)# R3.5.1
library(reshape2)
library(coda)
library(R.matlab)

source(file = "./BNJestimator/functionsourceAG.R")

# sample size
n <- 900
# MCMC Posterior sample size
m <- 2000

# Load data
INP <- read.table(file = "./datasets/x_AGcor_5000.txt")
OUT <- read.table(file = "./datasets/y_AGcor_5000.txt")


#Input index
kk <- 3 #3,10,18

x <- INP[1:n, kk]
y <- OUT[1:n, ]


## Domain vector of Y for trapezoidal numerical integration
# input
bwx <- 400 
xnew <- seq(from = (min(x)-0.01*(diff(range(x)))), 
            to= (max(x)+0.01*(diff(range(x)))), length.out = bwx)
# output
bwy <- 500
ynew <- seq(from = min(y)-0.01*diff(range(y)), 
            to= max(y)+0.01*diff(range(y)), length.out = bwy)



## MCMC
# Define Prior parameters
s2 <- matrix(c(var(x),0,0,var(y)),ncol=2) 
m2 <- c(mean(x),mean(y))
psiinv2 <- solve(s2)
prior <- list(a0=0.5,b0=0.5,nu1=4,nu2=4,s2=s2,m2=m2,psiinv2=psiinv2,tau1=1,tau2=1)
# Thin
th <- 1
state <- NULL
nsave <- m*th
nburn <- 1000
mcmc <- list(nburn=nburn,nsave=nsave,nskip=10,ndisplay=m)
# Posterior Joint distribution f(X,Y)
set.seed(1357)
fit.x <- DPdensity(y=cbind(x,y),prior=prior,mcmc=mcmc,state=state,status=TRUE,na.action=na.omit)
# function DPdensity is from library DPpackage
# package reference:
# Jara A, Hanson TE, Quintana FA, Müller P, Rosner GL. DPpackage: Bayesian semi-and nonparametric modeling in R. 
# Journal of statistical software. 2011 Apr 1;40(5):1.
randsave<-fit.x$save.state$randsave

# Extract MCMC posterior parameter samples
result.rand<-randsave[seq(1,nsave,by=th),]
mu.x<-as.matrix(result.rand[,seq(from=1,to=5*n,by=5)]) 
mu.y<-as.matrix(result.rand[,seq(from=2,to =5*n,by=5)])
sigma.x<-as.matrix(result.rand[,seq(from=3,to =5*n,by=5)])
sigma.xy<-as.matrix(result.rand[,seq(from=4,to =5*n,by=5)])
sigma.y<-as.matrix(result.rand[,seq(from=5,to =5*n,by=5)])
l_nc<-list();l_mux<-list();l_muy<-list();l_sigmax<-list();l_sigmaxy<-list();l_sigmay<-list(); #clear list
l_nc<-sapply(1:m,function(i) l_nc[[i]]<-count(mu.x[i,])[order(count(mu.x[i,])[,2]),][,2])
l_mux<-sapply(1:m,function(i)l_mux[[i]]<-count(mu.x[i,])[order(count(mu.x[i,])[,2]),][,1])
l_muy<-sapply(1:m, function(i) l_muy[[i]]<-count(mu.y[i,])[order(count(mu.y[i,])[,2]),][,1])
l_sigmax<-sapply(1:m, function(i) l_sigmax[[i]]<-count(sigma.x[i,])[order(count(sigma.x[i,])[,2]),][,1])
l_sigmaxy<-sapply(1:m, function(i) l_sigmaxy[[i]]<-count(sigma.xy[i,])[order(count(sigma.xy[i,])[,2]),][,1])
l_sigmay<-sapply(1:m, function(i) l_sigmay[[i]]<-count(sigma.y[i,])[order(count(sigma.y[i,])[,2]),][,1])


## $\delta$
Lowlim <- c( min(x)-0.001*diff(range(x)) , min(y)-0.001*diff(range(y)) )
Upplim <- c( max(x)+0.001*diff(range(x)), max(y)+0.001*diff(range(y)) )
D<-rep(0,m)
for (k in 1:m) {
  #R version 3.5, may take long
  D[k]<- divonne(f = Intdens, nComp = 1, lowerLimit=Lowlim, upperLimit = Upplim,
                 relTol = 1e-3,  absTol=1e-5, flags=list(verbose=0,final=0))$integral
}
D <- as.numeric(D)/2
# point estimate for $\delta$
delta <- mean(D)
# 95% credibility interval
quantile(D, probs = c(2.5, 97.5))


#KS
KSv<-rep(0,m)
for(k in 1:m){
  gX <- sapply(xnew,KSdis,ygrid=ynew)
  KSv[k]<-trapz(xnew,gX*mydensXT(xnew))
}
# point estimate for $\beta^{KS}$
KS <- mean(KSv,na.rm=T)
# 95% credibility interval
quantile(KSv, probs = c(0.025, 0.975),na.rm=T)

#eta
E <- rep(0,m)
for(k in 1:m){
  VEYX <- Vs(k)
  E[k] <- VEYX/VarY(k)
}
# point estimate for $\eta$
eta <- mean(E,na.rm=T)
# 95% credibility interval
quantile(E, probs = c(0.025, 0.975),na.rm=T)


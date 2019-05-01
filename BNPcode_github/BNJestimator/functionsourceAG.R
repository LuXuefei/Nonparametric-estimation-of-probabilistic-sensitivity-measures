###FUNCTIONS required for global sensitivity estimation ###

# True input distribution
mydensXT<-function(x) {dnorm(x, mean=1, sd = 1,log = FALSE)}


#k-iteration joint density evaluate at point xvals
mydens <- function(xvals){
  prop<-l_nc[[k]]/n
  l<-length(prop)
  l_sigmak<-list()
  l_sigmak<-lapply(1:l,function(j) matrix(c(l_sigmax[[k]][j],l_sigmaxy[[k]][j],l_sigmaxy[[k]][j],l_sigmay[[k]][j]),ncol=2))  
  mu.k<-cbind(l_mux[[k]], l_muy[[k]])
  result<-sapply(1:l, function(j) dmvnorm(xvals, mean = c(mu.k[j,]), sigma = l_sigmak[[j]], log = FALSE))%*% prop 
  return(result)} 

#k-iteration marginal density of Y, f_{Y}
mydensY <- function(y){
  prop<-l_nc[[k]]/n
  result<-sum(dnorm(y, mean=l_muy[[k]], sd=l_sigmay[[k]]^(.5))*prop)
  return( result )} 

## $\delta$
# Dispersion between joint and product of marginal
Intdens<-function(xvals){
  x_1<-xvals[1]
  x_2<-xvals[2]
  return(abs(mydensXT(x_1)*mydensY(x_2)-mydens(xvals)))
} #distance for k-th iteration



## For $beta^{KS}$
# l-th component: conditional expectation of Y|X, \nu_{l}^k
ExpYXl <- function(x, comp){
  result <- l_muy[[k]][comp]+l_sigmaxy[[k]][comp]*(1/l_sigmax[[k]][comp])*(x-l_mux[[k]][comp])
  return(result)
}
# l-th component: conditional variance of Y|X, \tau_{l}^k
VarYXl<-function(x,comp){
  result <- l_sigmay[[k]][comp]-l_sigmaxy[[k]][comp]*(1/l_sigmax[[k]][comp])*l_sigmaxy[[k]][comp]
  return(result)
}
# Conditional cdf at k-th iteration: F_Y|X
pnormmix.Fyx <- function(yt,xt) {
  lambda <- l_nc[[k]]/n
  nk <- length(lambda)
  pnorm.from.mix <- function(yt,comp) {
    lambda[comp]*pnorm(yt,mean=ExpYXl(xt,comp),
                            sd=VarYXl(xt,comp)^(.5))
  }
  pnorms <- sapply(1:nk,pnorm.from.mix,yt=yt)
  return(rowSums(pnorms))
}

# Marginal cdf at k-th iteration: F_Y
pnormmix.Fy <- function(y) {
  lambda <- l_nc[[k]]/n
  nk <- length(lambda)
  pnorm.from.mix <- function(y,component) {
    lambda[component]*pnorm(y,mean=l_muy[[k]][component],
                            sd=(l_sigmay[[k]][component])^(.5))
  }
  pnorms <- sapply(1:nk,pnorm.from.mix,y=y)
  return(rowSums(pnorms))
}

# Sup distance given value Xi=xt
KSdis<-function(ygrid,xt){
  result<-max(abs(pnormmix.Fy(ygrid)-pnormmix.Fyx(ygrid,xt))) 
  return(result)
}


## $\eta$
# \mu_{Y}^k(x) := E[Y|x,\theta]
comp0 <- function(xt){
  prop <- l_nc[[k]]/n
  result <- sum((l_sigmaxy[[k]]*(1/l_sigmax[[k]])*(xt-l_mux[[k]]))*prop) 
  return(result)}
# V^k: variance of \mu_{Y}^k(x)
Vs<- function(k){
  prop <- l_nc[[k]]/n
  comp <- sapply(xnew,function(xt) comp0(xt))
  result <- trapz(xnew,comp^2*mydensXT(xnew))
  return(result)
}
# Marginal variance of Y
VarY <-function(k){
  prop <- l_nc[[k]]/n
  nk<-length(prop)
  muy <- sum(l_muy[[k]]*prop)
  result <-sum(((l_muy[[k]]-rep(muy,nk))^2+l_sigmay[[k]])*prop)
  return(result)
}

#Trapezoidal Integration, values y at the points x
trapz <- function(x,y){
  idx = 2:length(x) 
  return (as.double( (x[idx] - x[idx-1]) %*% (y[idx] + y[idx-1])) / 2)}


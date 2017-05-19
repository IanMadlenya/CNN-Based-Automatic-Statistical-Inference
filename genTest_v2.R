#distributions: discreteRandom(9), Geometric(0.2), Exponential(5), Normal(3, 1), Poisson(7)
# each sample contains 900=30*30 datapoints
# each distribution contains 500 duplicates
#install.packages("LaplacesDemon")
#bernoulli, DiscreteUniform, geometric, negbinomial, exponential, normal, poisson)

library(LaplacesDemon)
library(rmutil)
library(smoothmest)
library(VGAM)
library(RMTstat)


args = commandArgs(trailingOnly=TRUE)
n = as.numeric(args[1])  # sample dimenstion, i.e. 30 * 30

target = matrix(, 0, 2)
input  = matrix(, 0, n)

#NN = 10 # as to distributions test
#num_rand = 100 # as to distributions test
num_rand = 100
NN = 1
factor = 1

label_id = 0
print("1, bernoulli") # continuous
for (parameter in runif(num_rand, min = 0.1, max = 0.9)) {
  input = rbind(input, matrix(rbern(NN*n, parameter), NN, n))
  #input = rbind(input,rbern(n, parameter)) # too slow
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("2, DiscreteUniform") # discrete
for (parameter in sample(2:11, num_rand, replace = TRUE)) {
  input = rbind(input, matrix(sample(1:parameter, NN*n,replace=T), NN, n))
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("3, geometric") # continuous
for (parameter in runif(num_rand, min = 0.1, max = 0.9)) {
  input = rbind(input, matrix(rgeom(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("4, negbinomial") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rnbinom(NN*n,parameter, 0.3), NN, n))
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("5, exponential") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rexp(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("6, normal") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rnorm(NN*n, parameter, 1), NN, n))
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("7, poisson") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rpois(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("8, beta") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rbeta(NN*n, parameter, 3), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("9, weibull") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rweibull(NN*n, parameter, 3), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("10, double_exp") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rdoublex(NN*n, parameter, 3), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("11, chi_square") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rchisq(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("12, f_distribution") # continuous
for (parameter in runif(num_rand, min = 3, max = 13)) {
  input = rbind(input, matrix(rf(NN*n, 3, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("13, gamma") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rgamma(NN*n, 0.5, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("14, logistic") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rlogis(NN*n, parameter, 0.5), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("15, lognorm") # continuous
for (parameter in runif(num_rand, min = -1, max = 2)) {
  input = rbind(input, matrix(rlnorm(NN*n, parameter, 0.5), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("16, pareto") # continuous
for (parameter in runif(num_rand, min = 1, max = 6)) {
  input = rbind(input, matrix(rpareto(NN*n, parameter, 2), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("17, t_distribution") # discrete
for (parameter in sample(4:14, num_rand, replace = TRUE)) {
  input = rbind(input, matrix(rt(NN*n, parameter, 2), NN, n) ) 
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("18, uniform") # continuous
for (parameter in runif(num_rand, min = 1, max = 5)) {
  input = rbind(input, matrix(runif(NN*n, parameter, parameter + 2), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("19, hypergeometric") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rhyper(NN*n, 3, parameter, 2), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("20, binomial") # discrete
for (parameter in sample(1:11, num_rand, replace = TRUE)) {
  input = rbind(input, matrix(rbinom(NN*n, parameter, 0.5), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("21, One-Inflated Logarithmic") # continuous
for (parameter in runif(num_rand, min = 0.1, max = 0.9)) {
  input = rbind(input, matrix(roilog(NN*n, parameter, pstr1 = 0), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("22, Triangle") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rtriangle(NN*n, parameter,lower=0,upper=10 * factor + 3), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("23, Distribution of the Wilcoxon Signed Rank Statistic") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rsignrank(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("24, Benini") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rbenini(NN*n, 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("25, Beta-Geometric") # continuous
for (parameter in runif(num_rand, min = 5, max = 15)) {
  input = rbind(input, matrix(rbetageom(NN*n, 5, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("26, Beta-Normal") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rbetanorm(NN*n, parameter, 10, mean = 5, sd = 11), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("27, Birnbaum-Saunders") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rbisa(NN*n, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("28, Dagum") # continuous
for (parameter in runif(num_rand, min = 5, max = 15)) {
  input = rbind(input, matrix(rdagum(NN*n, scale = 1, parameter,  2), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1


print("29, Frechet") # continuous
for (parameter in runif(num_rand, min = 5, max = 15)) {
  input = rbind(input, matrix(rfrechet(NN*n, location = 0, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1


print("30, Dirichlet") # continuous
for (parameter in runif(num_rand, min = 5, max = 15)) {
  input = rbind(input, matrix(rdiric(NN*n, shape = c(y1 = parameter, y2 = 2, y3 = 4)), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("31, Huber's Least Favourable") # continuous
for (parameter in runif(num_rand, min = 0.1, max = 1.5)) {
  input = rbind(input, matrix(rhuber(NN*n, k = parameter, mu = 0, sigma = 1), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("32, Gumbel") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rgumbel(NN*n, location = 0, scale = parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("33, Gompertz") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rgompertz(NN*n, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("34, Kumaraswamy") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rkumar(NN*n, 10, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("35, Laplace") # continuous, yet seemed to be sd rather than mean
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rlaplace(NN*n, location = 5, scale = parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("36, Log-Gamma") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rlgamma(NN*n, location = 0, scale = parameter, shape = 2), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("37, Lindley") # continuious
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rlind(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("38, Lomax") # continuous
for (parameter in runif(num_rand, min = 5, max = 15)) {
  input = rbind(input, matrix(rlomax(NN*n, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("39, Makeham") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rmakeham(NN*n, scale = 1, parameter, epsilon = 0), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("40, Maxwell") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rmaxwell(NN*n, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("41, Nakagami") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rnaka(NN*n, scale = 1, parameter, Smallno = 1.0e-6), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("42, Perks") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rperks(NN*n, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("43, Rayleigh") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rrayleigh(NN*n, scale = parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("44, Rice") # continuous
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rrice(NN*n, 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("45, Simplex") # continuous, tend to be sd rather than mean
for (parameter in runif(num_rand, min = 1, max = 11)) {
  input = rbind(input, matrix(rsimplex(NN*n, mu = 0.5, dispersion = parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("46, Singh-Maddala") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rsinmad(NN*n, scale = 1, 5, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("47, Skellam") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rskellam(NN*n, 5, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("48, Tobit") # continuous
for (parameter in runif(num_rand, min = 1, max = 10)) {
  input = rbind(input, matrix(rtobit(NN*n, mean = parameter, sd = 1, Lower = 0, Upper = Inf), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("49, Paralogistic") # continuous
for (parameter in runif(num_rand, min = 2, max = 12)) {
  input = rbind(input, matrix(rparalogistic(NN*n, scale = 1, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}
label_id = label_id + 1

print("50, The Zipf Distribution") # continuous
for (parameter in runif(num_rand, min = 1, max = 5)) {
  input = rbind(input, matrix(rzipf(NN * n, 10, parameter), NN, n) )
  tmp = matrix(, NN, 2); tmp[,1] = rep(parameter, each=NN); tmp[,2] = rep(label_id, each=NN)
  target = rbind(target, tmp)
}


dir = "/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/input/"

filename1 = paste(dir, 'test_input_50_', args[1], '.txt', sep='')
write.table(input, file=filename1, sep=',', col.names=FALSE, row.names=FALSE)

filename2 = paste(dir, 'test_target_50_', args[1], '.txt', sep='')
write.table(target, file=filename2, sep=',', col.names=FALSE, row.names=FALSE)




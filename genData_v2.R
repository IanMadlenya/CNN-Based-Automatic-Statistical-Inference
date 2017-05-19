#distributions: Bernouli(0.4), Geometric(0.2), Exponential(5), Normal(3, 1), Poisson(7)
# each sample contains 900=30*30 datapoints
# each distribution contains 500 duplicates

library(LaplacesDemon)
library(rmutil)
library(smoothmest)
library(VGAM)
library(RMTstat)

args = commandArgs(trailingOnly=TRUE)
NN = as.numeric(args[1]) # 1000
n = as.numeric(args[2]) # 30 * 30
factor = as.numeric(args[3])

# When there are too many distributions, the file-generating speed is extremely slow
# The solution is to seperate every distribution in a single file, and combine them to hdf5
# another advantage is making debugging easy


output <- function(id, feature, parameter, NN, n) {
  #dir = "/scratch/radon/d/deng106/CNNStatisticalModel/distributions/data/"
  #dir = "C:/Users/Wei/Desktop/"
  dir = args[4]
  #dir = "/scratch/radon/d/deng106/CNNStatisticalModel/distributions/data/900_sample_dimension/"
  featureFile = paste(dir, "feature_", n, "_", id, '.txt', sep='')
  write.table(feature, file=featureFile, sep=',', col.names=FALSE, row.names=FALSE)

  target = rbind(cbind(rep(parameter, each = NN), rep(id, NN*length(parameter))))
  labelFile = paste(dir, "label_", n, "_", id, '.txt', sep='')
  write.table(target, file=labelFile, sep=',', col.names=FALSE, row.names=FALSE)
}


print("1, bernoulli")
# mean = para
bernoulli = matrix(, 0, n)
parameter = seq(0.1, 0.9, 0.1/factor)
for (i in 1:length(parameter))
  bernoulli = rbind(bernoulli, matrix(rbern(NN*n, parameter[i]), NN, n)) 
output(0, bernoulli, parameter, NN, n)


print("2, DiscreteUniform")
# mean = (para + 1) / 2
DiscreteUniform = matrix(, 0, n)
parameter = seq(2, 11 * factor, 1)
for (i in 1:length(parameter))
  DiscreteUniform = rbind(DiscreteUniform, matrix(sample(1:parameter[i],NN*n,replace=T), NN, n))
output(1, DiscreteUniform, parameter, NN, n)



print("3, geometric")
# mean = 1/para
geometric = matrix(, 0, n)
parameter = seq(0.1, 0.9, 0.1/factor)
for (i in 1:length(parameter))
  geometric = rbind(geometric, matrix(rgeom(NN*n, parameter[i]), NN, n) )
output(2, geometric, parameter, NN, n)


print("4, negbinomial")
# mean = par1 * (1 - par2) / par2
negbinomial = matrix(, 0, n)
parameter = seq(1, 11 * factor, 1)
for (i in 1:length(parameter))
  negbinomial = rbind(negbinomial, matrix(rnbinom(NN*n,parameter[i],0.3), NN, n))
output(3, negbinomial, parameter, NN, n)



print("5, exponential")
# mean = para
exponential = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
  exponential = rbind(exponential, matrix(rexp(NN*n, parameter[i]), NN, n) )
output(4, exponential, parameter, NN, n)


print("6, normal")
# mean = para
normal = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
  normal = rbind(normal, matrix(rnorm(NN*n, parameter[i], 1), NN, n) )
output(5, normal, parameter, NN, n)



print("7, poisson")
# mean = para
poisson = matrix(, 0, n)
parameter = seq(1, 11 * factor, 1)
for (i in 1:length(parameter))
  poisson = rbind(poisson, matrix(rpois(NN*n, parameter[i]), NN, n) )
output(6, poisson, parameter, NN, n)


print("8, beta")
# mean = a / (a + b)
beta = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
beta = rbind(beta, matrix(rbeta(NN*n, parameter[i], 3), NN, n) )
output(7, beta, parameter, NN, n)


print("9, weibull")
# mean: b^(1/a) * gamma(1 + 1/a)
weibull = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
weibull = rbind(weibull, matrix(rweibull(NN*n, parameter[i], 3), NN, n) )
output(8, weibull, parameter, NN, n)



print("10, double_exp")
# mean: para
double_exp = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
double_exp = rbind(double_exp, matrix(rdoublex(NN*n, parameter[i], 3), NN, n) )
output(9, double_exp, parameter, NN, n)


print("11, chi_square")
# mean: para
chi_square = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
chi_square = rbind(chi_square, matrix(rchisq(NN*n, parameter[i]), NN, n) )
output(10, chi_square, parameter, NN, n)



print("12, f_distribution")
# mean: b / (b - 2)
f_distribution = matrix(, 0, n)
parameter = seq(3, 13, 1/factor)
for (i in 1:length(parameter))
f_distribution = rbind(f_distribution, matrix(rf(NN*n, 3, parameter[i]), NN, n) )
output(11, f_distribution, parameter, NN, n)


print("13, gamma")
# mean: a * b
gamma = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
gamma = rbind(gamma, matrix(rgamma(NN*n, 0.5, parameter[i]), NN, n) )
output(12, gamma, parameter, NN, n)


print("14, logistic")
# mean: para
logistic = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
logistic = rbind(logistic, matrix(rlogis(NN*n, parameter[i], 0.5), NN, n) )
output(13, logistic, parameter, NN, n)


print("15, lognorm")
# mean: exp(a + b^2/2)
loglnorm = matrix(, 0, n)
parameter = seq(-1, 2, 0.3/factor)
for (i in 1:length(parameter))
loglnorm = rbind(loglnorm, matrix(rlnorm(NN*n, parameter[i], 0.5), NN, n) )
output(14, loglnorm, parameter, NN, n)


print("16, pareto")
# mean: b* a / (b - 1)
pareto = matrix(, 0, n)
parameter = seq(1, 6, 0.5/factor)
for (i in 1:length(parameter))
pareto = rbind(pareto, matrix(rpareto(NN*n, parameter[i], 2), NN, n) )
output(15, pareto, parameter, NN, n)


print("17, t_distribution")
# mean: 0
t_distribution = matrix(, 0, n)
parameter = seq(4, 14 * factor, 1)
for (i in 1:length(parameter))
t_distribution = rbind(t_distribution, matrix(rt(NN*n, parameter[i], 2), NN, n) )
output(16, t_distribution, parameter, NN, n)


print("18, uniform")
# mean: (a + b) / 2
uniform = matrix(, 0, n)
parameter = seq(1, 5, 0.5/factor)
for (i in 1:length(parameter))
uniform = rbind(uniform, matrix(runif(NN*n, parameter[i], parameter[i] + 2), NN, n) )
output(17, uniform, parameter, NN, n)


print("19, hypergeometric")
# mean: c * b / a, a must be bigger than c
hypergeometric = matrix(, 0, n)
parameter = seq(1, 11 * factor, 1)
for (i in 1:length(parameter))
hypergeometric = rbind(hypergeometric, matrix(rhyper(NN*n, 3, parameter[i], 2), NN, n) )
output(18, hypergeometric, parameter, NN, n)


print("20, binomial")
# mean: nn * a, here a = p belong to (0,1)
binomial = matrix(, 0, n)
parameter = seq(1, 11 * factor, 1)
for (i in 1:length(parameter))
binomial = rbind(binomial, matrix(rbinom(NN*n, parameter[i], 0.5), NN, n) )
output(19, binomial, parameter, NN, n)




print("21, One-Inflated Logarithmic")
distribution = matrix(, 0, n)
parameter = seq(0.1, 0.9, 0.1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(roilog(NN*n, parameter[i], pstr1 = 0), NN, n) )
output(20, distribution, parameter, NN, n)


print("22, Triangle")
distribution = matrix(, 0, n)
parameter = seq(1, 10 * factor, 1)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rtriangle(NN*n, parameter[i],lower=0,upper=10 * factor + 3), NN, n) )
output(21, distribution, parameter, NN, n)


## new
print("23, Distribution of the Wilcoxon Signed Rank Statistic")
distribution = matrix(, 0, n)
parameter = seq(1, 10 * factor, 1)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rsignrank(NN*n, parameter[i]), NN, n) )
output(22, distribution, parameter, NN, n)


print("24, Benini")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rbenini(NN*n, 1, parameter[i]), NN, n) )
output(23, distribution, parameter, NN, n)

print("25, Beta-Geometric")
distribution = matrix(, 0, n)
parameter = seq(5, 15 * factor, 1)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rbetageom(NN*n, 5, parameter[i]), NN, n) )
output(24, distribution, parameter, NN, n)

print("26, Beta-Normal")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rbetanorm(NN*n, parameter[i], 10, mean = 5, sd = 11), NN, n) )
output(25, distribution, parameter, NN, n)

print("27, Birnbaum-Saunders")
distribution = matrix(, 0, n)
parameter = seq(1, 10, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rbisa(NN*n, scale = 1, parameter[i]), NN, n) )
output(26, distribution, parameter, NN, n)

print("28, Dagum")
distribution = matrix(, 0, n)
parameter = seq(5, 15, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rdagum(NN*n, scale = 1, parameter[i],  2), NN, n) )
output(27, distribution, parameter, NN, n)

print("29, Frechet")
distribution = matrix(, 0, n)
parameter = seq(5, 15, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rfrechet(NN*n, location = 0, scale = 1, parameter[i]), NN, n) )
output(28, distribution, parameter, NN, n)


##### new 
print("30, Dirichlet")
distribution = matrix(, 0, n)
parameter = seq(5, 15, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rdiric(NN*n, shape = c(y1 = parameter[i], y2 = 2, y3 = 4)), NN, n) )
output(29, distribution, parameter, NN, n)

print("31, Huber's Least Favourable")
distribution = matrix(, 0, n)
parameter = seq(0.1, 1.5, 0.15/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rhuber(NN*n, k = parameter[i], mu = 0, sigma = 1), NN, n) )
output(30, distribution, parameter, NN, n)
#### 

print("32, Gumbel")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rgumbel(NN*n, location = 0, scale = parameter[i]), NN, n) )
output(31, distribution, parameter, NN, n)


print("33, Gompertz")
distribution = matrix(, 0, n)
parameter = seq(1, 10, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rgompertz(NN*n, scale = 1, parameter[i]), NN, n) )
output(32, distribution, parameter, NN, n)


print("34, Kumaraswamy")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rkumar(NN*n, 10, parameter[i]), NN, n) )
output(33, distribution, parameter, NN, n)


print("35, Laplace")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rlaplace(NN*n, location = 5, scale = parameter[i]), NN, n) )
output(34, distribution, parameter, NN, n)


print("36, Log-Gamma")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rlgamma(NN*n, location = 0, scale = parameter[i], shape = 2), NN, n) )
output(35, distribution, parameter, NN, n)


print("37, Lindley")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rlind(NN*n, parameter[i]), NN, n) )
output(36, distribution, parameter, NN, n)


print("38, Lomax")
distribution = matrix(, 0, n)
parameter = seq(5, 15, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rlomax(NN*n, scale = 1, parameter[i]), NN, n) )
output(37, distribution, parameter, NN, n)

print("39, Makeham")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rmakeham(NN*n, scale = 1, parameter[i], epsilon = 0), NN, n) )
output(38, distribution, parameter, NN, n)

print("40, Maxwell")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rmaxwell(NN*n, parameter[i]), NN, n) )
output(39, distribution, parameter, NN, n)


print("41, Nakagami")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rnaka(NN*n, scale = 1, parameter[i], Smallno = 1.0e-6), NN, n) )
output(40, distribution, parameter, NN, n)


print("42, Perks")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rperks(NN*n, scale = 1, parameter[i]), NN, n) )
output(41, distribution, parameter, NN, n)

print("43, Rayleigh")
distribution = matrix(, 0, n)
parameter = seq(1, 10, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rrayleigh(NN*n, scale = parameter[i]), NN, n) )
output(42, distribution, parameter, NN, n)


print("44, Rice")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rrice(NN*n, 1, parameter[i]), NN, n) )
output(43, distribution, parameter, NN, n)


print("45, Simplex")
distribution = matrix(, 0, n)
parameter = seq(1, 11, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rsimplex(NN*n, mu = 0.5, dispersion = parameter[i]), NN, n) )
output(44, distribution, parameter, NN, n)


print("46, Singh-Maddala")
distribution = matrix(, 0, n)
parameter = seq(1, 10, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rsinmad(NN*n, scale = 1, 5, parameter[i]), NN, n) )
output(45, distribution, parameter, NN, n)

print("47, Skellam")
distribution = matrix(, 0, n)
parameter = seq(1, 10 * factor, 1)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rskellam(NN*n, 5, parameter[i]), NN, n) )
output(46, distribution, parameter, NN, n)

print("48, Tobit")
distribution = matrix(, 0, n)
parameter = seq(1, 10, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rtobit(NN*n, mean = parameter[i], sd = 1, Lower = 0, Upper = Inf), NN, n) )
output(47, distribution, parameter, NN, n)

print("49, Paralogistic")
distribution = matrix(, 0, n)
parameter = seq(2, 12, 1/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rparalogistic(NN*n, scale = 1, parameter[i]), NN, n) )
output(48, distribution, parameter, NN, n)

if (0) {
  print("cauchy")
  # mean: does not exist
  cauchy = matrix(, 0, n)
  parameter = seq(1, 11, 1/factor)
  for (i in 1:length(parameter))
  cauchy = rbind(cauchy, matrix(rcauchy(NN*n, parameter[i], 3), NN, n) )
  output(49, cauchy, parameter, NN, n)
}

print("50, The Zipf Distribution")
distribution = matrix(, 0, n)
parameter = seq(1, 5, 0.5/factor)
for (i in 1:length(parameter))
distribution = rbind(distribution, matrix(rzipf(NN * n, 10, parameter[i]), NN, n) )
output(49, distribution, parameter, NN, n)

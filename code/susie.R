#install.packages("susieR")
library(susieR)
data(N3finemapping)
saveRDS(N3finemapping, "../data/N3finemapping.rds")

b <- true_coef[,1]
y

fitted <- susie(X, Y[,1],
                L = 10,
                verbose = TRUE)

print(fitted$sets)

sets <- susie_get_cs(fitted,
                     X = X,
                     coverage = 0.9,
                     min_abs_corr = 0.1)



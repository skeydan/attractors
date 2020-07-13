
library(nonlinearTseries)
library(reticulate)

# correlation dimension <=  information <= capacity (box counting)
# corr dim gives dimensionality of attractor


ds <- py_load_object("data/geyser_train_test.pkl")
ds <- ds[1:10000]

# tau-delay estimation based on the autocorrelation function
tau.acf = timeLag(ds, technique = "acf",
                  lag.max = 100, do.plot = T)
# tau-delay estimation based on the mutual information function
tau.ami = timeLag(ds, technique = "ami", 
                  lag.max = 100, do.plot = T)

emb.dim = estimateEmbeddingDim(ds, time.lag = tau.ami, max.embedding.dim = 15)

emb.dim
tau.ami

tak = buildTakens(ds,embedding.dim = emb.dim, time.lag = tau.ami)

cd = corrDim(ds,
             min.embedding.dim = emb.dim,
             max.embedding.dim = emb.dim + 5,
             time.lag = tau.ami, 
             min.radius = 0.001, max.radius = 50,
             n.points.radius = 40,
             do.plot=FALSE)
plot(cd)
cd.est = estimate(cd, regression.range=c(5,17),
                  use.embeddings = 8:12)

se = sampleEntropy(cd, do.plot = F)
se.est = estimate(se, do.plot = F,
                  regression.range = c(5,17))
mean(se.est)


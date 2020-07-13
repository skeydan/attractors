

library(nonlinearTseries)
library(reticulate)

ds <- py_load_object("data/geyser_train_test.pkl")
ds <- py_load_object("data/electricity_train_test.pkl")
ds <- py_load_object("data/ecg_train.pkl")
ds <- py_load_object("data/mouse.pkl")



ds <- ds[1:10000]

# correlation dim <= information dim <= capacity dim (box counting)


# https://www.sciencedirect.com/science/article/abs/pii/0167278983901252
# The relevant definitions of dimension are of two general types, those that depend only on metric properties,
# and those that depend on the frequency with which a typical trajectory visits different regions of the attractor.
# ... the conclusion that a all of the frequency dependent dimensions take on the same value,
# which we call the “dimension of the natural measure”, and all of the metric dimensions take on a common value,
# which we call the “fractal dimension”.
# Furthermore, the dimension of the natural measure is typically equal to the Lyapunov dimension

# see also: https://www.scielo.br/scielo.php?script=sci_arttext&pid=S0100-73862001000400004

# embedding ---------------------------------------------------------------

# tau-delay estimation based on the mutual information function
tau.ami = timeLag(ds,
                  technique = "ami",
                  lag.max = 100,
                  do.plot = F)

emb.dim = estimateEmbeddingDim(ds, time.lag = tau.ami, max.embedding.dim = 15)

emb.dim
tak = buildTakens(ds, embedding.dim = emb.dim, time.lag = tau.ami)



# correlation dimension ---------------------------------------------------

# correlation dimension <=  information <= capacity (box counting)
# corr dim gives dimensionality of attractor

cd = corrDim(
  ds,
  min.embedding.dim = emb.dim,
  max.embedding.dim = emb.dim + 5,
  time.lag = tau.ami,
  #min.radius = 0.001,
  #max.radius = 50,
  min.radius = 0.99,
  max.radius = 3,
  n.points.radius = 40,
  do.plot = FALSE
)
plot(cd)

cd.est = estimate(cd,
                  #regression.range = c(5, 17),
                  #use.embeddings = 8:12
                  #regression.range = c(20, 30),
                  #use.embeddings = 10:13
                  #regression.range = c(5, 15),
                  #use.embeddings = 13:16
                  regression.range = c(1, 2),
                  use.embeddings = 12:16
                  )

cd.est
# geyser 1.7
# elec 6.0
# ecg 3.69
# mouse 0.15



# entropy -----------------------------------------------------------------


se = sampleEntropy(cd, do.plot = F)
se.est = estimate(se, do.plot = F,
                  regression.range = c(5, 17))
mean(se.est)

# geyser 0.22
# elec 0.41
# ecg 0.2
# mouse nan





# max Ljapunov exponent ---------------------------------------------------


# λ < 0
# The orbit attracts to a stable fixed point or stable periodic orbit.
# Negative Lyapunov exponents are characteristic of dissipative or non-conservative systems
# (the damped harmonic oscillator for instance). Such systems exhibit asymptotic stability;
# the more negative the exponent, the greater the stability. 
#
# λ = 0
# The orbit is a neutral fixed point (or an eventually fixed point).
# A Lyapunov exponent of zero indicates that the system is in some sort of steady state mode.
# A physical system with this exponent is conservative. 
#
# λ > 0
# The orbit is unstable and chaotic. Nearby points, no matter how close, will diverge to any arbitrary separation.
# All neighborhoods in the phase space will eventually be visited. These points are said to be unstable.


# get the sampling period
# computing the differences of time (all differences should be equal)
sampling.period = diff(ds)
ml = maxLyapunov(
  ds,
  sampling.period = 0.01,
  min.embedding.dim = emb.dim,
  max.embedding.dim = emb.dim + 3,
  time.lag = tau.ami,
  radius = 1,
  max.time.steps = 1000,
  do.plot = FALSE
)
plot(ml, type = "l", xlim = c(0, 8))
ml.est = estimate(
  ml,
  regression.range = c(0, 3),
  do.plot = T,
  type = "l"
)
ml.est

# geyser 0.27
# elec 0
# ecg 0
# mouse  -0.1



# surrogate data testing --------------------------------------------------

st = surrogateTest(
  ds,
  significance = 0.05,
  one.sided = F,
  FUN = timeAsymmetry,
  do.plot = F
)

plot(st)

# geyser reject
# elec accept
# ecg reject
# mouse accept

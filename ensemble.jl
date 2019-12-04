# # Simple example of a homogeneous ensemble using learning networks

# In this simple example, no bagging is used, so every atomic model
# gets the same learned parameters, unless the atomic model training
# algorithm has randomness, eg, DecisionTree with random subsampling
# of features at nodes.

using MLJ
using Plots; pyplot()

# learning network (composite model spec):

Xs = source()
ys = source(kind=:target)

atom = @load DecisionTreeRegressor
atom.n_subfeatures = 4 # to ensure diversity among trained atomic models

machines = (machine(atom, Xs, ys) for i in 1:100)

# overload summation for nodes:
Base.sum(v...) = sum(v)
Base.sum(v::AbstractVector{<:AbstractNode}) = node(sum, v...)

yhat = sum([predict(m, Xs) for  m in machines]);


# new composite model type and instance:

one_hundred_models = @from_network OneHundredModels(atom=atom) <= yhat

# application to data

X, y = @load_boston;

# tune regularization parameter for a *single* tree:

r = range(atom,
          :min_samples_split,
          lower=2,
          upper=100, scale=:log)

mach = machine(atom, X, y)

curve = learning_curve!(mach,
                        range=r,
                        measure=mav,
                        resampling=CV(nfolds=9))

plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name)

# tune regularization parameter for all trees in ensemble simultaneously:

r = range(one_hundred_models,
          :(atom.min_samples_split),
          lower=2,
          upper=100, scale=:log)

mach = machine(one_hundred_models, X, y)

curve = learning_curve!(mach,
                        range=r,
                        measure=mav,
                        resampling=CV(nfolds=9))

plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name)

#-

using Literate #src
Literate.notebook(@__FILE__, @__DIR__) #src

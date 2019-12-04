# Simple example of a homogeneous ensemble using learning networks

using MLJ

# learning network (composite model spec):

Xs = source()
ys = source(kind=:target)

atom = @load DecisionTreeRegressor
atom.n_subfeatures = 4

machines = (machine(atom, Xs, ys) for i in 1:100)

# overload summation for nodes:
Base.sum(v...) = sum(v)
Base.sum(v::AbstractVector{<:AbstractNode}) = node(sum, v...)

yhat = sum([predict(m, Xs) for  m in machines]);


# new composite model type and instance:

one_hundred_trees = @from_network OneHundredTrees(atom=atom) <= yhat

X, y = @load_boston;
evaluate(one_hundred_trees, X, y, measure=rms)

using Literate #src
Literate.notebook(@__FILE__, @__DIR__) #src

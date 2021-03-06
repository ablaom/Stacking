# ## DEFINING A COMPOSITE MODEL FOR MODEL STACKING BY HAND

# In stacking one blends the predictions of different regressors or
# classifiers to gain, in some cases, better performance than naive
# averaging or majority vote.

# Here we illustrate how to build a two-model stack as an MLJ learning
# network, which we export as a new stand-alone composite model
# type `MyTwoStack`. This will make the stack that we build completely
# re-usable (new data, new models) and means we can apply
# meta-algorithms, such as performance evaluation and tuning to the
# stack, exaclty as we would for any other model.

# Our main purpose is to demonstrate the flexibility of MLJ's
# composite model interface. Eventually, MLJ will provide built-in
# composite types or macros to achieve the same results in a few lines,
# which will suffice for routine stacking tasks.

# After exporting our learning network as a composite model, we
# instantiate the model for an application to the Ames House Price data
# set.


# ### Basic stacking using out-of-sample base learner predictions

# A rather general stacking protocol was first described in a [1992
# paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
# by David Wolpert. For a generic introduction to the basic two-layer
# stack described here, see [this blog
# post](https://burakhimmetoglu.com/2016/12/01/stacking-models-for-improved-predictions/)
# of Burak Himmetoglu.

# A basic stack consists of a number of base learners (two, in this
# illustration) and single adjudicating model.

# When a stacked model is called to make a prediction, the individual
# predictions of the base learners are made the columns of an *input*
# table for the adjudicating model, which then outputs the final
# prediction. However, it is crucial to understand that the flow of
# data *during training* is not the same.

# The base model predictions used to train the adjudicating model are
# *not* the predictions of the base learners fitted to all the
# training data. Rather, to prevent the adjudicator giving too much
# weight to the base learners with low *training* error, the input
# data is first split into a number of folds (as in cross-validation),
# a base learner is trained on each fold complement individually, and
# corresponding predictions on the folds are spliced together to form
# a full-length out-of-sample prediction. For illustrative purposes we
# use just three folds. Each base learner will get three separate
# machines, for training on each fold complement, and a fourth
# machine, trained on all the supplied data, for use in the prediction
# flow.

# We build the learning network with dummy data at the source nodes,
# so the reader can experiment with the network as it is built (by
# calling `fit!` on nodes, and by calling the nodes themselves, as
# they are defined). As usual, this data is not seen by the exported
# composite model type, and the component models we choose are just
# default values for the hyperparameters of the composition model.

using MLJ
using Plots
pyplot(size=(200*1.5, 120*1.5))
import Random.seed!
seed!(1234)

# Some models we will need:

linear = @load LinearRegressor pkg=MLJLinearModels
ridge = @load RidgeRegressor pkg=MultivariateStats; ridge.lambda = 0.01
knn = @load KNNRegressor; knn.K = 4
tree = @load DecisionTreeRegressor; min_samples_leaf=1
forest = @load RandomForestRegressor; forest.n_estimators=500

# ### Warm-up exercise: Define a model type to average predictions

# Let's define a composite model type `MyAverageTwo` that
# averages the predictions of two deterministic regressors. Here's the learning network:

X = source()
y = source(kind=:target)

model1 = linear
model2 = knn

m1 = machine(model1, X, y)
y1 = predict(m1, X)

m2 = machine(model2, X, y)
y2 = predict(m2, X)

yhat = 0.5*y1 + 0.5*y2

# And the macro call to define `MyAverageTwo` and an instance `average_two`:

avg = @from_network MyAverageTwo(regressor1=model1,
                                       regressor2=model2) <= yhat

# Evaluating this average model on the Boston data set, and comparing
# with the base model predictions:

evaluate(linear, (@load_boston)..., measure=rms)
evaluate(knn, (@load_boston)..., measure=rms)
evaluate(avg, (@load_boston)..., measure=rms)


# ### Step 0: Helper functions:

# To generate folds:

folds(data, nfolds) =
    partition(1:nrows(data), (1/nfolds for i in 1:(nfolds-1))...);

# For example, we have:
f = folds(1:10, 3) 

# In our learning network, the folds will depend on the input data,
# which will be wrapped as a source node. We therefore need to
# overload the `folds` function for nodes:

folds(X::AbstractNode, nfolds) = node(XX -> folds(XX, nfolds), X);

# It will also be convenient to use the MLJ method `restrict(X, f, i)`
# that restricts data `X` to the `i`th element (fold) of `f`, and
# `corestrict(X, f, i)` that restricts to the corresponding fold
# complement (the concatenation of all but the `i`th
# fold).

# For example, we have:

corestrict(string.(1:10), f, 2)

# Overloading these functions for nodes:

MLJ.restrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> restrict(XX, ff, i), X, f);
MLJ.corestrict(X::AbstractNode, f::AbstractNode, i) =
    node((XX, ff) -> corestrict(XX, ff, i), X, f);

# All the other data manipulations we will need (`vcat`, `hcat`,
# `MLJ.table`) are already overloaded to work with nodes.


# ### Step 1: Choose some test data (optional) and some component models (defaults for the composite model):

steps(x) = x < -3/2 ? -1 : (x < 3/2 ? 0 : 1)
x = Float64[-4, -1, 2, -3, 0, 3, -2, 1, 4]
Xraw = (x = x, )
yraw = steps.(x);
plt = plot(steps, xlim=(-4.5, 4.5), label="truth");
scatter!(deepcopy(plt), x, yraw, label="data")

# Some models to stack:

model1 = linear
model2 = knn

# The adjudicating model:

judge = linear


# ### Step 2: Define the training nodes

# Let's instantiate some input and target source nodes for the
# learning network, wrapping the play data defined above:

# Wrapped as source node:

X = source(Xraw)
y = source(yraw; kind=:target)

# Our first internal node represents the three folds (vectors of row
# indices) for creating the out-of-sample predictions:

f = folds(X, 3)
f()

# Constructing machines for training `model1` on each fold-complement:

m11 = machine(model1, corestrict(X, f, 1), corestrict(y, f, 1))
m12 = machine(model1, corestrict(X, f, 2), corestrict(y, f, 2))
m13 = machine(model1, corestrict(X, f, 3), corestrict(y, f, 3))

# Define each out-of-sample prediction of `model1`:

y11 = predict(m11, restrict(X, f, 1));
y12 = predict(m12, restrict(X, f, 2));
y13 = predict(m13, restrict(X, f, 3));

# Splice together the out-of-sample predictions for model1:

y1_oos = vcat(y11, y12, y13);

# Optionally, to check our network so far, we can fit and plot
# `y1_oos`:

fit!(y1_oos, verbosity=0)
scatter!(deepcopy(plt), x, y1_oos(), label="linear oos")

# We now repeat the procedure for the other model:

m21 = machine(model2, corestrict(X, f, 1), corestrict(y, f, 1))
m22 = machine(model2, corestrict(X, f, 2), corestrict(y, f, 2))
m23 = machine(model2, corestrict(X, f, 3), corestrict(y, f, 3))
y21 = predict(m21, restrict(X, f, 1));
y22 = predict(m22, restrict(X, f, 2));
y23 = predict(m23, restrict(X, f, 3));

# And testing the knn out-of-sample prediction:

y2_oos = vcat(y21, y22, y23);
fit!(y2_oos, verbosity=0)
scatter!(deepcopy(plt), x, y2_oos(), label="knn oos")

# Now that we have the out-of-sample base learner predictions, we are
# ready to merge them into the adjudicator's input table and construct
# the machine for training the adjudicator:

X_oos = MLJ.table(hcat(y1_oos, y2_oos))
m_judge = machine(judge, X_oos, y)

# Are we done with constructing machines? Well, not quite. Recall that
# when use the stack to make predictions on new data, we will be
# feeding the adjudicator ordinary predictions on the base
# learners. But so far, we have only defined machines to train the
# base learners on fold complements, not on the full data, which we do
# now:

m1 = machine(model1, X, y)
m2 = machine(model2, X, y)


# ### Step 3: Define nodes still needed for prediction

# To obtain the final prediction, `yhat`, we get the base learner
# predictions, based on training with all data, and feed them to the
# adjudicator:
y1 = predict(m1, X);
y2 = predict(m2, X);
X_judge = MLJ.table(hcat(y1, y2))
yhat = predict(m_judge, X_judge)

# Let's check the final prediction node can be fit and called:
fit!(yhat, verbosity=0)
scatter!(deepcopy(plt), x, yhat(), label="yhat")

# Although of little statistical significance here, we note that
# stacking gives a lower *training* error than naive averaging:

e1 = rms(y1(), y())
e2 = rms(y2(), y())
emean = rms(0.5*y1() + 0.5*y2(), y())
estack = rms(yhat(), y())
@show e1 e2 emean estack;


# ### Step 4: Export the learning network as a new model type

# The learning network (less the data wrapped in the source nodes)
# amounts to a specification of a new composite model type for
# two-model stacks, trained with three-fold resampling of base model
# predictions. Let's create the new type `MyTwoModelStack` (and an
# instance):

instance = @from_network MyTwoModelStack(regressor1=model1,
                                         regressor2=model2,
                                         judge=judge) <= yhat

# And this completes the definition of our re-usable stacking model type.


# ### Applying `MyTwoModelStack` to Ames House Price data

# Without undertaking any hyperparameter optimization, we evaluate the
# performance of a random forest and ridge regressor on the well-known
# Ames House Prices data, and compare the performance of a stack
# using the random forest and ridge regressors as base learners.

# #### Data pre-processing

# Here we use a reduced subset of the Ames House Price data set with
# 12 features:

X0, y0 = @load_reduced_ames;

# Inspect scitypes:

function sci(X)
    s = schema(X)
    (names=collect(s.names), scitypes=collect(s.scitypes)) |> pretty
end

sci(X0)

# Coerce counts and ordered factors to continuous:

X1 = coerce(X0, :OverallQual => Continuous,
            :GarageCars => Continuous,
            :YearRemodAdd => Continuous,
            :YearBuilt => Continuous);

# One-hot encode the multiclass:

hot_mach = fit!(machine(OneHotEncoder(), X1))
X = transform(hot_mach, X1);

# Check the final scitype:

scitype(X)

# transform the target:

y1 = log.(y0)
y = transform(fit!(machine(UnivariateStandardizer(), y1)), y1);

# #### Define the stack and compare performance:

avg = MyAverageTwo(regressor1=forest,
                   regressor2=ridge)


stack = MyTwoModelStack(regressor1=forest,
                        regressor2=ridge,
                        judge=ridge)

forest.n_estimators = 200
forest.min_samples_split = 3
forest.max_features=4

evaluate(forest, X, y, measure=rms)

#-

evaluate(linear, X, y, measure=rms)

#-

evaluate(avg, X, y, measure=rms)

#-

evaluate(stack, X, y, measure=rms)



# r = range(forest, :n_estimators, lower=20, upper=1000)
# mach = machine(forest, X, y)
# curves = learning_curve!(mach, range=r, n=4)
# plot(curves.parameter_values, curves.measurements, xlab=curves.parameter_name)

using Literate #src
Literate.notebook(@__FILE__, @__DIR__) #src

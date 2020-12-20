module DSUtils

using LinearAlgebra
using FreqTables
using NamedArrays

using Reexport
@reexport using StatsBase
@reexport using DataFrames
@reexport using Plots

export BCDiag, bcdiag
export biasplot, ksplot, rocplot, accuracyplot
export liftcurve, cumliftcurve
export liftable, cumliftable

export onehot, onehot!

export ranks                # equal count binning
export strstd               # string standardization
export sumxm                # element-wise sum of vectors with missing as 0

export kstest               # two sample KS and location
export auroc                # concordance and AUROC
export concordance          # Concordance with bounds function

export infovalue

include("bcdiag.jl")
include("onehotencoder.jl")
include("utils.jl")

end # module

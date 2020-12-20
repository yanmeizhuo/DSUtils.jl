"""
    ranks(x; groups = 10, rank = tiedrank, rev = false)

Return a variable which bins `x` into `groups` number of bins.
The `rank` keyword allows different ranking method;
use `rev = true` to reverse sort so that small bin number is large value of `x`.
Missing values are assigned to group `missing`.

Default values of `rank = tiedrank` and `rev = false` results
in similar grouping as SAS PROC RANK groups=n tied=mean.
"""
function ranks(x::Vector{T} where {T<:Real}; groups::Int = 10, rank = tiedrank, rev = false)::Vector{Int32}
    nx = length(x)
    nx >= 2 || throw(ArgumentError("x length should be >= 2"))
    groups >= 2 || throw(ArgumentError("groups should be >= 2"))

    floor.(Int32, rank(x, rev = rev) * groups / (nx+1))
end

function ranks(x::Vector{T} where {T<:Union{Missing,Real}}; groups::Int = 10, rank = tiedrank, rev = false)::Vector{Union{Missing,Int32}}
    nx = length(x)
    nx >= 2 || throw(ArgumentError("x length should be >= 2"))
    groups >= 2 || throw(ArgumentError("groups should be >= 2"))

    floor.(Union{Missing,Int32}, rank(x, rev = rev) * groups / (nx+1))
end


"""
    strstd(s; empty = "")

Standardizes string `s` by stripping leading and trailing blanks,
embedded blanks are replaced with underscore _.
If `s` is missing or all blanks, the result is value of `empty`.

Default of `empty` is "", missing values and all blank strings become "".
"""
function strstd(s::AbstractString; empty = "")
    str = strip(s)                          # strip leading and trailing blanks
    if isempty(str)                         # replace empty string
        str = empty                         # empty is not subject to blank to underscore
    else
        str = replace(str, ' ' => '_')        # embedded blanks become underscore
    end

    return str
end

function strstd(s::Missing; empty = "")
    return empty
end


"""
    sumxm(x...)     sum treating missings as 0

Returns the sum of `x` treating missing values as 0's.
`x` can be varying number of scalars, in this case, their sum is returned.
If `x` is varying number of vectors, they are summed element-wise across the vectors.
"""
sumxm(x...) = _sumxm.(x...)                         # . over "rows" across varying vectors
_sumxm(x::T...) where T<:Real = sum(x)              # no missing
_sumxm(x...) = sum(z -> ismissing(z) ? 0 : z, x)    # missing is 0 in sum


"""
    kstest(class, var; rev = true)

Calculate empirical 2 sample Kolmogorov-Smirnov statistic and its location.
`class` is a 2 level categorical variable, `var` is the distribution to analyze.

Returns:
- n, total number of observations
- n1, number of observations of class 1
- n0, number of observations of class 0
- baserate, incidence rate of class 1
- ks, the maximum separation between the two cumulative distributions
- ksarg, the value of `var` at which maximum separation is achieved
- ksdep, depth of ksarg in the sorted values of `var`;
`rev` = true counts depth from high value towards low value.
"""
function kstest(class::BitArray{1}, var::Vector; rev = true)
    n = length(class)
    n == length(var) || error(ArgumentError("class and var should have the same length"))
    n1 = sum(class)
    n1 == 0 && error(ArgumentError("there are no class 1"))
    n0 = n - n1
    n0 == 0 && error(ArgumentError("there are no class 0"))

    baserate = n1 / n
    idx = sortperm(var, rev = rev)
    tgt = class[idx]
    cdf1 = cumsum(tgt) ./ n1            # TPR: True Positive Rate
    cdf0 = cumsum(.!tgt) ./ n0          # FPR: False Positive Rate
    sep = cdf1 .- cdf0
    ks, ksidx = findmax(sep)
    ksarg = var[idx[ksidx]]             # var @ max sep
    ksdep = ksidx / n                   # depth @ max sep

    (n = n, n1 = n1, n0 = n0, baserate = baserate, ks = ks, ksarg = ksarg, ksdep = ksdep)
end

function kstest(class::Vector, var::Vector; rev = true)
    uc = unique(class)
    length(uc) == 2 || error(ArgumentError("class should have 2 levels"))

    kstest(class .== uc[2], var; rev = rev)
end


"""
    auroc(class, var; tie = 1e-6)

Calculate area under Receiver Operating Characteristics (ROC) curve,
`class` is a 2 level categorical variable, `var` is the distribution to analyze.
Pair-wise comparison between class 1 values with class 0 values are made as follows:
- class 1 value > class 0 value is Concordant
- class 1 value ≈ class 0 value (within `tie`) is Tied
- class 1 value < class 0 value is Discordant

Returns:
- concordant, number of concordant comparisons
- tied, number of tied comparisons
- discordant, number of discordant comparisons
- auc, or C, is (Concordant + 0.5Tied) / Total comparisons; same as numeric integration of ROC curve
- gini, 2C-1, also known as Somer's D, is (Concordant - Discordant) / Total comparisons

Note there are other rank order measures:
- Goodman-Kruskal Gamma is (Concordant - Discordant) / (Concordant + Discordant), no penalty for ties
- Kendall's Tau is (Concordant - Discordant) / (0.5 N(N-1))
"""
function auroc(class::BitArray{1}, var::Vector; tie = 1e-6)
    n = length(class)
    n == length(var) || error(ArgumentError("class and var should have the same length"))
    n1 = sum(class)
    n1 == 0 && error(ArgumentError("there are no class 1"))
    n0 = n - n1
    n0 == 0 && error(ArgumentError("there are no class 0"))

    c1 = sort!(var[class])              # sorted var of class 1
    c0 = sort!(var[.!class])            # sorted var of class 0

    conc = 0
    tied = 0
    l_ix = 1
    u_ix = 1
    @inbounds for v in c1               # loop over this usually smaller array
        l_v = v - tie
        u_v = v + tie

        while l_ix <= n0                # first index within [l_v, u_v] window
            l_v <= c0[l_ix] && break
            l_ix += 1                   # could exit as n0 + 1
        end

        while u_ix <= n0                # first index beyond [l_v, u_v] window
            u_v < c0[u_ix] && break
            u_ix += 1                   # could exit as n0 + 1
        end

        conc += (l_ix - 1)              # lower than l_ix are Concordant
        tied += (u_ix - l_ix)           # within window are tied
    end

    pairs = n1 * n0
    disc = pairs - conc - tied
    auc = (conc + 0.5tied) / pairs

    (conc = conc, tied = tied, disc = disc, auc = auc, gini = 2auc - 1)
end

function auroc(class::Vector, var::Vector; tie = 1e-6)
    uc = unique(class)
    length(uc) == 2 || error(ArgumentError("class should have 2 levels"))

    auroc(class .== uc[2], var; tie = tie)
end


"""
    concordance(class, var, tie)

Concordance calculation with flexible tied region (`auroc` uses fixed width region).
`class` is a 2 level categorical variable, `var` is the distribution to analyze,
`tie`(x) returns the lower and upper bound of tied region of x.

Pair-wise comparison between class 1 values with class 0 values are made as follows:
class 1 value > class 0 value is Concordant; class 1 value ≈ class 0 value (within `tie`) is Tied;
class 1 value < class 0 value is Discordant.

Returns:
- concordant, number of concordant comparisons
- tied, number of tied comparisons
- discordant, number of discordant comparisons
- auroc, or C, is (Concordant + 0.5Tied) / Total comparisons; same as numeric integration of ROC curve
- gini, 2C-1, also known as Somer's D, is (Concordant - Discordant) / Total comparisons

Note Goodman-Kruskal Gamma is (Concordant - Discordant) / (Concordant + Discordant);
and Kendall's Tau is (Concordant - Discordant) / (0.5 x Total count x (Total count - 1))
"""
function concordance(class::BitArray{1}, var::Vector, tie)
    n = length(class)
    n == length(var) || error(ArgumentError("class and var should have the same length"))
    n1 = sum(class)
    n1 == 0 && error(ArgumentError("there are no class 1"))
    n0 = n - n1
    n0 == 0 && error(ArgumentError("there are no class 0"))

    c1 = sort!(var[class])              # sorted var of class 1
    c0 = sort!(var[.!class])            # sorted var of class 0

    conc = 0
    tied = 0
    l_ix = 1
    u_ix = 1
    @inbounds for v in c1               # loop over this usually smaller array
        l_v, u_v = tie(v)               # <== tie function

        while l_ix <= n0                # first index within [l_v, u_v] window
            l_v <= c0[l_ix] && break
            l_ix += 1                   # could exit as n0 + 1
        end

        while u_ix <= n0                # first index beyond [l_v, u_v] window
            u_v < c0[u_ix] && break
            u_ix += 1                   # could exit as n0 + 1
        end

        conc += (l_ix - 1)              # lower than l_ix are Concordant
        tied += (u_ix - l_ix)           # within window are tied
    end

    pairs = n1 * n0
    disc = pairs - conc - tied
    auc = (conc + 0.5tied) / pairs

    (conc = conc, tied = tied, disc = disc, auc = auc, gini = 2auc - 1)
end

function concordance(class::Vector, var::Vector, tie)
    uc = unique(class)
    length(uc) == 2 || error(ArgumentError("class should have 2 levels"))

    concordance(class .== uc[2], var, tie)
end


"""
    infovalue(g::Vector{Integer}, b::Vector{Integer}))

Information value calculation of `g`, `b` vector of binned frequency counts

- weight of evidence = log(density g / density b), 0 adjusted
- infovalue = sum (density g - density b) * weight of evidence

Industry rule of thumb:
- iv <= 0.1         no significant change
- 0.1 < iv <= 0.25  minor change
- 0.25 < iv         major change
"""
function infovalue(g::Vector{TG} where TG <: Integer,
                   b::Vector{TB} where TB <: Integer; woeadjust = 0.5)
    length(g) >= 2 || error("frequency vectors should have at least 2 elements")
    length(g) == length(b) || error("frequency vectors should be the same length")

    # infovalue is class invariant - swapping g/b doesn't matter
    # infovalue is order invariant - re-ordering rows doesn't matter
    # implementation modelled after SAS PROC BINNING where
    # weight of evidence is adjusted to avoid divide by 0

    pb = b ./ sum(b)
    pg = g ./ sum(g)
    pba = (b + (b .== 0) * woeadjust) ./ sum(b)     # adjust only pure bins
    pga = (g + (g .== 0) * woeadjust) ./ sum(g)
    woe = log.(pga ./ pba)
    sum((pg .- pb) .* woe)
end

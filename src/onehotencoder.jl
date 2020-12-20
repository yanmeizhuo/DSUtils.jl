"""
    onehot(lvl, x::Vector)

Create an indicator variable of `x` equal to `lvl`.
`lvl` and `x` can have missing values,
`ismissing` is used where necessary to ensure correct result.
"""
function onehot(lvl::Missing, x::Vector{T}) where T >: Missing
    return ismissing.(x)
end

function onehot(lvl::Missing, x::Vector)
    return zeros(Bool, size(x))
end

function onehot(lvl::T, x::Vector{Union{Missing,T}}) where T
    out = zeros(Bool, size(x))
    for (i,v) in enumerate(x)
        if !ismissing(v)
            out[i] = (v == lvl)
        end
    end

    return out
end

onehot(lvl::T, x::Vector{T}) where T = (x .== lvl)


"""
    onehot!(df::AbstractDataFrame, var::Symbol)

One hot encode unique values of `var` in `df`.
New variable name is constructed as `var_lvl`.
If `df.var` is string, it is standardized first via `strstd` before one hot encoding.
"""
function onehot!(df::AbstractDataFrame, var::Symbol)
    if eltype(df[!, var]) <: AbstractString
        vec = strstd(df[!, var])
    else
        vec = df[!, var]
    end

    for lvl in sort!(unique(vec))
        var_lvl = Symbol(var, '_', lvl)
        println("... creating $var_lvl")
        df[!, var_lvl] = onelvl(lvl, vec)
    end
end


"""
    onehot!(df::AbstractDataFrame, vars::Vector{Symbol})

One hot encode variables in `vars` in `df`.
"""
function onehot!(df::AbstractDataFrame, vars::Vector{Symbol})
    for var in unique(vars)
        onehot!(df, var)
    end
end

"""
    BCDiag

A structure of diagnostic properties of a binary Classifier.
Facilitates summary plots and tables.
"""
struct BCDiag
    n::Int                  # Number of observations
    n1::Int                 # Number of Class 1
    n0::Int                 # Number of Class 0
    baserate::Float64       # Class 1 incidence rate
    ks::Float64             # Maximum separation
    ksarg::Float64          # max sep occurs at this predictor value
    ksdep::Float64          # max sep occurs at this depth

    concordant::Int         # concordant pairs
    tied::Int               # tied pairs
    discordant::Int         # discordant pairs
    auc::Float64            # Area under ROC = (concordant + 0.5tied)/ total pairs
    gini::Float64           # Gini = 2C - 1

    grp::Vector             # group number [0-groups]
    depth::Vector           # Depth [0, 1]
    cdf1::Vector            # cum1 / n1 = True Positive Rate
    cdf0::Vector            # cum0 / n0 = False Positive Rate

    cntg::Vector            # count this group
    cnt1::Vector            # count class 1 this group
    prd1::Vector            # predicted class 1 this group
    rrObs::Vector           # response rate actual
    rrPrd::Vector           # response rate Predicted

    cumg::Vector            # cum count up to this group
    cum1::Vector            # cum count class 1 up to this group
    cpr1::Vector            # cum predicted class 1 up to this group
    crObs::Vector           # cum response rate actual
    crPrd::Vector           # cum response rate predicted
end

function Base.show(io::IO, ::MIME"text/plain", x::BCDiag)
    print(io,
        "Base rate: ", round(x.baserate, digits = 4),"   n: ", x.n, "   n1: ", x.n1, "   n0: ", x.n0, "\n",
        "ks:        ", round(x.ks, digits = 4), "   occurs at value of ", x.ksarg, " depth of ", x.ksdep, "\n",
        "auroc:     ", round(x.auc, digits = 4), "   concordant: ", x.concordant,
        "   tied: ", x.tied, "   discordant: ", x.discordant, "\n",
        "Gini:      ", round(x.gini, digits = 4)
    )
end

function Base.show(io::IO, x::BCDiag)
    print(io, round(x.ks, digits = 4), "   ", round(x.auc, digits = 4) )
end


"""
    bcdiag(target, pred; groups = 100, rev = true, tie = 1e-6)

Perform diagnostics of a binary classifier.
`target` is a 2 level categorical variable, `pred` is probability of class 1.
`groups` is the number of bins to use for plotting/printing.
`rev` = true orders `pred` from high to low.
`tie` is the tolerance of `pred` where values are considered tied.

Returns a BCDiag struct which can be used for plotting or printing:
- `biasplot` is calibration plot of `target` response rate vs. `pred` response rate
- `ksplot` produces ksplot of cumulative distributions
- `rocplot` plots the Receiver Operating Characteristics curve
- `accuracyplot` plots the accuracy curve with adjustable utility
- `liftcurve` is the lift curve
- `cumliftsurve` is the cumulative lift surve
- `liftable` is the lift table as a DataFrame
- `cumliftable` is the cumulative lift table as a DataFrame
"""
function bcdiag(target::BitArray{1}, pred::Vector; groups = 100, rev = true, tie = 1e-6)
    ks = kstest(target, pred; rev = rev)
    roc = auroc(target, pred; tie = tie)

    grpid = ranks(pred, groups = groups, rev = rev)
    frqtbl = freqtable(grpid, target)
    frq = frqtbl.array
    grp = NamedArrays.names(frqtbl, 1)

    n0, n1 = sum(frq, dims = 1)
    n = n0 + n1

    cntg = vec(sum(frq, dims = 2))          # bin count
    cnt1 = @view frq[:, 2]                  # bin count class 1
    cnt0 = @view frq[:, 1]                  # bin count class 0
    cumg = cumsum(cntg)                     # cum bin count
    cum1 = cumsum(cnt1)                     # cum class 1
    cum0 = cumsum(cnt0)                     # cum class 0

    depth = cumg ./ n                       # Depth including this group
    cdf1  = cum1 ./ n1                      # TPR: True Positive Rate
    cdf0  = cum0 ./ n0                      # FPR: False Positive Rate

    rrObs = cnt1 ./ cntg                    # actual response rate
    crObs = cum1 ./ cumg                    # cum actual response rate

    prd1 = Vector{Float64}(undef, length(grp))  # predicted class 1
    for (i, g) in enumerate(grp)
        prd1[i] = sum(pred[grpid .== g])
    end
    cpr1 = cumsum(prd1)                     # cum predicted class 1

    rrPrd = prd1 ./ cntg                    # predicted response rate
    crPrd = cpr1 ./ cumg                    # cum predicted response rate

    BCDiag( ks..., roc..., grp, depth, cdf1, cdf0,
            cntg, cnt1, prd1, rrObs, rrPrd, cumg, cum1, cpr1, crObs, crPrd )
end

function bcdiag(target::Vector, pred::Vector; groups = 100, rev = true, tie = 1e-6)
    uc = unique(target)
    length(uc) == 2 || error(ArgumentError("target should have 2 levels"))

    bcdiag(target .== uc[2], pred; rev = rev)
end


"""
    biasplot(x::BCDiag)

returns a bias calibration plot of `x` - actual response vs. predicted response
"""
function biasplot(x::BCDiag)
    dmin = min(minimum(x.rrPrd), minimum(x.rrObs))
    dmax = max(maximum(x.rrPrd), maximum(x.rrObs))

    plt = plot(size = (500, 500), aspect_ratio = :equal, legend = :bottomright,
        xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("Model Calibration Graph", titlefontsize = 11)
    xlabel!("Predicted Response Rate", xguidefontsize = 10)
    ylabel!("Observed Response Rate", yguidefontsize = 10)

    plot!(x.rrPrd, x.rrObs, label = nothing, seriescolor = :blue)
    plot!([dmin, dmax], [dmin, dmax], label = nothing, width = 0.5, seriescolor = :black)

    plt
end


"""
    ksplot(x::BCDiag)

returns a KS plot of `x` - CDF1 (True Positive) and CDF0 (False Positive) versus depth
"""
function ksplot(x::BCDiag)
    plt = plot(size = (500, 500), aspect_ratio = :equal, xlims = (0., 1.), ylims = (0., 1.),
        legend = :bottomright, xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("KS Graph", titlefontsize = 11)
    xlabel!("Depth", xguidefontsize = 10)
    ylabel!("Empirical CDF", yguidefontsize = 10)

    plot!([0.; x.depth], [0.; x.cdf1], label = "Class 1", width = 1.5, seriescolor = :blue)
    plot!([0.; x.depth], [0.; x.cdf0], label = "Class 0", width = 0.5, seriescolor = :black)
    plot!([0., x.baserate, 1.], [0., 1., 1.], label = "Perfect Class 1", width = 0.5, seriescolor = :blue, linestyle = :dash)
    plot!([0., x.baserate, 1.], [0., 0., 1.], label = "Perfect Class 0", width = 0.5, seriescolor = :black, linestyle = :dash)
    ksidx = findlast(x.depth .<= x.ksdep)
    plot!([x.depth[ksidx], x.depth[ksidx]], [x.cdf0[ksidx], x.cdf1[ksidx]], label = nothing, width = 0.8, linestyle = :dash)

    annotate!((0.03, 0.97, text("Base rate = $(round(x.baserate, digits = 4))", :black, :left, 9)))
    annotate!((0.03, 0.93, text("Depth = $(round(x.ksdep, digits = 4))", :black, :left, 9)))
    annotate!((0.03, 0.89, text("KSarg = $(round(x.ksarg, digits = 4))", :black, :left, 9)))
    annotate!((0.03, 0.85, text("KS = $(round(x.ks, digits = 4))", :black, :left, 9)))

    plt
end


"""
    rocplot(x::BCDiag)

returns a ROC plot of `x` - CDF1 (True Positive) vs. CDF0 (False Positive)
"""
function rocplot(x::BCDiag)
    plt = plot(size = (500, 500), aspect_ratio = :equal, xlims = (0., 1.), ylims = (0., 1.),
        legend = :bottomright, xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("ROC Graph", titlefontsize = 11)
    xlabel!("False Positive Rate", xguidefontsize = 10)
    ylabel!("True Positive Rate", yguidefontsize = 10)

    plot!([0.; x.cdf0], [0.; x.cdf1], label = "Model", seriescolor = :blue, width = 1.5)
    plot!([0., 1.], [0., 1.], label = "Random", seriescolor = :black, width = 0.3, linestyle = :dash)

    annotate!((0.75, 0.22, text("auc = $(round(x.auc, digits = 4))", :black, :left, 9)))
    annotate!((0.75, 0.18, text("Gini = $(round(2. * x.auc - 1., digits = 4))", :black, :left, 9)))

    plt
end


"""
    accuracyplot(x::BCDiag; util=[1, 0, 0, 1])

Using `util` values for [TP, FN, FP, TN], produce accuracy plot and its [max, argmax, argdep].
Default `util` values of [1, 0, 0, 1] gives the standard accuracy value of (TP+TN)/N.
"""
function accuracyplot(x::BCDiag; util = [1, 0, 0, 1])
    p = x.baserate
    q = 1. - p

    # perfect classifier
    up = dot([p, 0., 0., q], util)                  # TP, FN, FP, TN of perfect classifier

    # 0.5 / 0.5 random classifier
    ur5 = dot([0.5p, 0.5p, 0.5q, 0.5q], util)

    # baserate random classifier
    urb = dot([p*p, p*q, p*q, q*q], util)

    # model utility value [TP, FN, FP, TN]
    uc = [p.*x.cdf1  p.*(1. .- x.cdf1)  q.*x.cdf0  q.*(1. .- x.cdf0)] * util
    ucmax = findmax(uc)

    plt = plot(size = (500, 500), xlims = (0., 1.), legend = :best,
        xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("Accuracy Plot", titlefontsize = 11)
    xlabel!("Depth", xguidefontsize = 10)
    ylabel!("Utility", yguidefontsize = 10)

    hline!([up], label = "Perfect $(round(up, sigdigits = 3))", linestyle = :dash)

    plot!(x.depth, uc, width = 1.5, seriescolor = :blue,
        label = "Model $(round(ucmax[1], sigdigits = 3))")
    vline!([x.depth[ucmax[2]]], label = "Depth $(x.depth[ucmax[2]])")

    hline!([urb], label = "Base Random $(round(urb, sigdigits = 3))", linestyle = :dash)

    plt
end


"""
    liftcurve(x::BCDiag)

returns a lift curve plot of `x` - actual and predicted versus depth
"""
function liftcurve(x::BCDiag)
    plt = Plots.plot(size = (500, 500), xlims = (0., 1.),
        legend = :topright, xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("Lift Curve", titlefontsize = 11)
    xlabel!("Depth", xguidefontsize = 10)
    ylabel!("Response Rate", yguidefontsize = 10)

    plot!(x.depth, x.rrObs, label = "Actual", width = 0.5, linestyle = :dash)
    plot!(x.depth, x.rrPrd, label = "Predicted", width = 1.5, seriescolor = :blue)
    hline!([x.baserate], label = "Base rate", linestyle = :dash)

    plt
end


"""
    cumliftcurve(x::BCDiag)

returns a cumulative lift curve plot of `x` - cumulative actual and predicted vs. depth
"""
function cumliftcurve(x::BCDiag)
    plt = Plots.plot(size = (500, 500), xlims = (0., 1.),
        legend = :topright, xguidefontsize = 10, yguidefontsize = 10, titlefontsize = 11)

    title!("Cumulative Lift Curve", titlefontsize = 11)
    xlabel!("Depth", xguidefontsize = 10)
    ylabel!("Response Rate", yguidefontsize = 10)

    plot!(x.depth, x.crObs, label = "Actual", width = 0.5, linestyle = :dash)
    plot!(x.depth, x.crPrd, label = "Predicted", width = 1.5, seriescolor = :blue)
    hline!([x.baserate], label = "Base rate", linestyle = :dash)

    plt
end


"""
    liftable(x::BCDiag)

returns a lift table of `x` as a DataFrame
"""
function liftable(x::BCDiag)
    out = DataFrame(grp = x.grp, depth = x.depth, count = x.cntg,
                    cntObs = x.cnt1, cntPrd = x.prd1,
                    rrObs = x.rrObs, rrPred = x.rrPrd,
                    liftObs = x.rrObs ./ x.baserate, liftPrd = x.rrPrd ./ x.baserate
                    )
end


"""
    cumliftable(x::BCDiag)

returns a cumulative lift table of `x` as a DataFrame
"""
function cumliftable(x::BCDiag)
    out = DataFrame(grp = x.grp, depth = x.depth, count = x.cumg,
                    cumObs = x.cum1, cumPrd = x.cpr1,
                    crObs = x.cum1 ./ x.cumg,
                    crPrd = x.cpr1 ./ x.cumg,
                    )
    out.liftObs = out.crObs ./ x.baserate
    out.liftPrd = out.crPrd ./ x.baserate

    out
end

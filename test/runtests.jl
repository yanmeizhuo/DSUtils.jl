using DSUtils
using Test

@testset "Data Science Utilities" begin
    @testset "ranks binning" begin
        x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        xm = [1, 1, missing, 2, 2, 2, 3, 3, 3, 3]
        @test ranks(x) == [1, 1, 1, 5, 5, 5, 5, 5, 8, 8]
        @test ranks(x, rev=true) == [8, 8, 8, 4, 4, 4, 4, 4, 1, 1]
        @test ranks(x, rank=StatsBase.ordinalrank) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        @test ranks(x, rank=StatsBase.competerank) == [0, 0, 0, 3, 3, 3, 3, 3, 8, 8]
        @test ranks(x, rank=StatsBase.denserank) == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        @test isequal(ranks(xm), [1, 1, missing, 3, 3, 3, 6, 6, 6, 6])
    end

    @testset "strstd string standardization" begin
        @test strstd("abc") == "abc"
        @test strstd("  abc def  ") == "abc_def"
        @test strstd("") == ""
        @test strstd("   ") == ""
        @test strstd(missing) == ""

        @test strstd("abc", empty="0") == "abc"
        @test strstd("  abc def  ", empty="0") == "abc_def"
        @test strstd("", empty="0") == "0"
        @test strstd("   ", empty="0") == "0"
        @test strstd(missing, empty="0") == "0"

        @test strstd("abc", empty=missing) == "abc"
        @test strstd("  abc def  ", empty=missing) == "abc_def"
        @test ismissing(strstd("", empty=missing))
        @test ismissing(strstd("   ", empty=missing))
        @test ismissing(strstd(missing, empty=missing))
    end

    @testset "onehot encoder" begin
        x = [1, 2, 3, 1, 2, 3, 4, 4, 4]
        xm = [1, 2, 3, 1, 2, 3, 4, 4, 4, missing, missing]
        s = ["a", "b", "c"]
        sm = ["a", "b", "c", missing]

        @test onehot(1, x) == [1, 0, 0, 1, 0, 0, 0, 0, 0]
        @test onehot(2, x) == [0, 1, 0, 0, 1, 0, 0, 0, 0]
        @test onehot(5, x) == [0, 0, 0, 0, 0, 0, 0, 0, 0]
        @test onehot(missing, x) == [0, 0, 0, 0, 0, 0, 0, 0, 0]

        @test onehot(1, xm) == [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        @test onehot(2, xm) == [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        @test onehot(5, xm) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        @test onehot(missing, xm) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

        @test onehot("a", s) == [1, 0, 0]
        @test onehot("b", s) == [0, 1, 0]
        @test onehot(" ", s) == [0, 0, 0]
        @test onehot(missing, s) == [0, 0, 0]

        @test onehot("a", sm) == [1, 0, 0, 0]
        @test onehot("b", sm) == [0, 1, 0, 0]
        @test onehot(" ", sm) == [0, 0, 0, 0]
        @test onehot(missing, sm) == [0, 0, 0, 1]
    end

    @testset "sumxm sum with missings as 0" begin
        v1 = [1, 1, 1, 1, 1, 1]
        v2 = [2, 2, 2, 2, 2, 2]
        vm = [3, 3, 3, missing, missing, missing]

        @test sumxm(1, 2, 3) == 6
        @test sumxm(1, 2, 3, missing) == 6

        @test sumxm(v1, v2) == [3, 3, 3, 3, 3, 3]
        @test sumxm(v1, v2, vm) == [6, 6, 6, 3, 3, 3]
        @test sumxm(vm, vm) == [6, 6, 6, 0, 0, 0]
    end

    @testset "ks roc                      " begin
        t = [0, 0, 0, 1, 1, 1]
        x = [1, 1, 1, 2, 2, 2]

        @test kstest(t, x) == (n = 6, n1 = 3, n0 = 3, baserate = 0.5, ks = 1.0, ksarg = 2, ksdep = 0.5)
        @test auroc(t, x) == (conc = 9, tied = 0, disc = 0, auc = 1.0, gini = 1.0)
    end

end

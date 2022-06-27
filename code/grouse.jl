function GROUSE(M::HePPCATModel,Y::Matrix{Float64},Yidx::AbstractMatrix,step::Float64,stat_fxn::Function,err_fxn::Function)

    stats = [stat_fxn(M)]
    errs = [err_fxn(M)]
    times = []
    d,n = size(Y)
    data_idx = randperm(n)
    for t = 1:n
        j = data_idx[t]
        telapsed = @elapsed begin
            streamGROUSE!(M,Y[:,j],Yidx[:,j],step)
        end
        push!(stats,stat_fxn(M))
        push!(errs,err_fxn(M))
        push!(times,telapsed)
    end

    return M, stats, errs, times
end



function streamGROUSE!(M::HePPCATModel, v::Vector{Float64}, xIdx::AbstractVector, step::Float64)

        ### Main GROUSE update
        U = M.U
        ΩIdx = findall(>(0), xIdx)
        U_Omega = U[ΩIdx,:]
        v_Omega = v[ΩIdx]
        w_hat = pinv(U_Omega) * v_Omega

        r = v_Omega - U_Omega * w_hat

        rnorm = norm(r)
        wnorm = norm(w_hat)
        sigma = rnorm * norm(w_hat)

        if step > 0
            t = step
        else
            t = atan(rnorm / wnorm)
        end

        if t < π / 2
            alpha = (cos(t) - 1) / wnorm^2
            beta = sin(t) / sigma
            Ustep = U * (alpha * w_hat)
            Ustep[ΩIdx] = Ustep[ΩIdx] + beta * r
            M.U .= U + Ustep * w_hat'
        else
            M.U .= U
        end

        return M

end
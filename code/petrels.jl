# function OnlineHPPCA(M,Y,ΩY,groups,λ,δ)
#     d,k = size(M.F)
#     n = size(Y)[2]
#     Rₜ⁺ = [δ*Matrix(I(k)) for i=1:d]
#     Yrec = deepcopy(Y)
#     data_idx = randperm(n)
#     stats_log = [stats_fcn(M)]
#     err_log = [Fmeasure(M)]
#     time_log = []
#     for t = 1:n
# #         j = data_idx[t]

#         yₜ = Y[:,t]
#         l = groups[t]
#         Ωₜ = ΩY[:,t]
#         telapsed = @elapsed begin
#         yₜʳ = streamHPPCA!(M,yₜ,l,Ωₜ,Rₜ⁺,λ)
#         end
#         Yrec[:,t] = yₜʳ
#         push!(stats_log,stats_fcn(M))
#         push!(err_log, Fmeasure(M))
#         push!(time_log,telapsed)
#     end
#     return M, Yrec, stats_log, err_log, time_log
# end

function PETRELS(M,Y,ΩY,λ,δ)
    d,k = size(M.F)
    n = size(Y)[2]
    Rₜ⁺ = [δ*Matrix(I(k)) for i=1:d]
    Yrec = deepcopy(Y)
    data_idx = randperm(n)
    stats_log = [stats_fcn(M)]
    err_log = [Fmeasure(M)]
    time_log = []
    for t = 1:n
#         j = data_idx[t]
        yₜ = Y[:,t]
        Ωₜ = ΩY[:,t]
        telapsed = @elapsed begin
        M,Rₜ⁺,yₜʳ = streamPETRELS!(M,yₜ,Ωₜ,Rₜ⁺,λ)
        end
        Yrec[:,t] = yₜʳ
        push!(stats_log,stats_fcn(M))
        push!(err_log, Fmeasure(M))
        push!(time_log,telapsed)
    end
    return M, Yrec, stats_log, err_log, time_log
end

function streamPETRELS!(M,yₜ,Ωₜ,Rₜ⁺,λ)
    Ωₜidx = findall(>(0), Ωₜ)
    F = copy(M.F)
    FΩₜ = F[Ωₜidx,:]
    yΩₜ = yₜ[Ωₜidx]
    zₜ = pinv(FΩₜ)*yΩₜ 
    Z = zₜ * zₜ'
    updatefparams!.(Ωₜ,Rₜ⁺,Ref(λ),Ref(Z))
     # updatefparams!.(Ωₜ,Rₜ⁺,Ref(λ),Ref(zₜ))
    F[Ωₜidx,:] = hcat(updatef!.(eachrow(FΩₜ),yΩₜ,Rₜ⁺[Ωₜidx],Ref(zₜ))...)'
    Fhat = svd(F)
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt
    
    yₜʳ = M.F * zₜ

    return M,Rₜ⁺,yₜʳ
end

function updatefparams!(Ωₜ,Rₜ⁺,λ,Z)

     Rₜ⁺ .= Ωₜ*Z + λ*Rₜ⁺

     return Rₜ⁺
end

# function updatefparams!(Ωₜ,Rₜ⁺,λ,zₜ)

#     βₜ = 1 + 1/λ*zₜ'*Rₜ⁺*zₜ 
#     vₜ = 1/λ*Rₜ⁺*zₜ 
#     Rₜ⁺ .= 1/λ*Rₜ⁺ - Ωₜ*1/βₜ*vₜ*vₜ'

#     return Rₜ⁺
# end

function updatef!(f,yₜ,Rₜ⁺,zₜ)

    f .+= (yₜ - f'*zₜ)*inv(Rₜ⁺)*zₜ
    # f .+= (yₜ - f'*zₜ)*Rₜ⁺*zₜ
    return f
end

# function updatef!(f,yₜ,Ωₜ,Rₜ⁺,λ,zₜ,vℓ)
# #     βₜ = 1 + 1/λ*zₜ'*Rₜ⁺*zₜ 
# #     vₜ = 1/λ*Rₜ⁺*zₜ 
# #     Rₜ⁺ .= 1/λ*Rₜ⁺ - Ωₜ/vℓ*1/βₜ*vₜ*vₜ'
#     Rₜ⁺ .= Ωₜ/vℓ*zₜ*zₜ' + λ*Rₜ⁺
# #     f .+= Ωₜ/vℓ*(yₜ - f'*zₜ)*Rₜ⁺*zₜ
#     f .+= Ωₜ/vℓ*(yₜ - f'*zₜ)*inv(Rₜ⁺)*zₜ
#     return f
# end
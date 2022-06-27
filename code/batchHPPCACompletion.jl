function batchHPPCACompletion(M::HePPCATModel,Y::Matrix{Float64},ΩY::AbstractMatrix,niters::Int64,Fmeasure::Function,stats_fcn::Function,buffer=1)


	stats_log = []
	err_log = []

	stats_log = [stats_fcn(M)]
    err_log = [Fmeasure(M)]
	L = length(Y)
	n = [size(Yl,2) for Yl in Y]
	Ωidx = [[findall(>(0), ΩY[l][:,i]) for i=1:n[l]] for l=1:L]

	θ = [sum([length(Ωidx[l][i]) for i=1:n[l]]) for l=1:L]
	α = [sum([norm(Y[l][:,i][Ωidx[l][i]])^2 for i=1:n[l]]) for l=1:L]

    time_log = [0.]
    tlast = 0
	for iter = 1:niters

		telapsed = @elapsed begin
			M, Mt, zt = updateF!(M,Y,ΩY,Ωidx)
			M = updatev!(M,Y,Ωidx,Mt,zt,θ,α)
		end
		tlast += telapsed
		if(mod(iter,buffer)==0)
	        push!(stats_log,stats_fcn(M))
	        push!(err_log, Fmeasure(M))
	        push!(time_log,tlast)
	        tlast = 0
	    end

    end

	return M, stats_log, err_log, time_log
end


function updateF!(M,Y,Ω,Ωidx)
	F  = M.F
	k = size(F,2)
	v = M.v
	L = length(Y)
	n = [size(Yl,2) for Yl in Y]

	### Compute the surrogate parameterrs
	Mt = []
	zt = []
	for l=1:L 
		Mtl = []
		ztl = []
		for i=1:n[l]
			Ωli = Ωidx[l][i]
			FΩ = F[Ωli,:]
			yΩ = Y[l][:,i][Ωli]
			Mtli = inv(FΩ'*FΩ + v[l]*I)
			push!(Mtl, inv(FΩ'*FΩ + v[l]*I))
			push!(ztl, Mtli*FΩ'*yΩ)
		end
		push!(Mt, Mtl)
		push!(zt, ztl)
	end

	### Update F

	Fhat = hcat(computefj.(Ref(M),Ref(Y),Ref(Ω),Ref(Mt),Ref(zt),1:d)...)'

	Fhat = svd(Fhat)
    
    M.U .= Fhat.U
    M.λ .= Fhat.S.^2
    M.Vt .= Fhat.Vt

    return M, Mt, zt

end


function computefj(M,Y,Ω,Mt,zt,j)
	k = size(M.F,2)
	R = zeros(k,k)
	s = zeros(k)
	L = length(Y)
	n = [size(Yl,2) for Yl in Y]

	for l=1:L
		for i=1:n[l]
			Ωlij = Ω[l][:,i][j]
			if(Ωlij > 0)
				R += 1/v[l] * (zt[l][i]*zt[l][i]' + v[l]*Mt[l][i])
				s += 1/v[l] * Y[l][:,i][j]*zt[l][i]
			end
		end
	end

	return inv(R)*s
end

function updatev!(M,Y,Ωidx,Mt,zt,θ,α)
	F = M.F
	v = M.v

	for l=1:L
		# θl = 0
		# αl = 0
		ρtl = 0
		for i=1:n[l]
			Ωli = Ωidx[l][i]
			FΩ = F[Ωli,:]
			yΩ = Y[l][:,i][Ωli]
			ztli = zt[l][i]

			# θl += length(Ωli)
			# αl += norm(yΩ)^2
			ρtl += 2*yΩ'*(FΩ*ztli) - (norm(FΩ*ztli)^2 + v[l]*tr((FΩ'*FΩ)*Mt[l][i]))

		end

		M.v[l] = (α[l] - ρtl) / θ[l]
	end

	return M

end
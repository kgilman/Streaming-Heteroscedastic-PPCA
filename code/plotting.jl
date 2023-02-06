function plottraces(stats,labels,colors,markers,interval::Int64,alpha,figsize,fontsize,title::String,xlabel::String,legend=:outerright,yaxis=:linear)
  
    p = plot(size=figsize,legend=legend,title=title,xlabel=xlabel,yaxis=yaxis,
        xtickfont=font(fontsize), 
        ytickfont=font(fontsize), 
        guidefont=font(fontsize), 
        legendfont=font(fontsize),
        titlefont=font(fontsize))
    for (stat,label) in zip(stats,labels)
        plot!(p,[(1:interval:length(trace),trace[1:interval:length(trace)]) for trace in stat],alpha=alpha,color=colors[label],label="")
        # plot!(p,1:interval[label]:length(mean(stat)),mean(stat)[1:interval[label]:length(mean(stat))],color=colors[label],linestyle=:dash,markershape=markers[label],width=2,label=label)
        plot!(p,1:interval:length(mean(stat)),mean(stat)[1:interval:length(mean(stat))],color=colors[label],linestyle=:dash,markershape=markers[label],width=2,label=label)
    end
    
    return p
end


function plottraces(stats,labels,colors,markers,interval::Dict{String, Int64},alpha,figsize,fontsize,title::String,xlabel::String,legend=:outerright,yaxis=:linear)
  
    p = plot(size=figsize,legend=legend,title=title,xlabel=xlabel,yaxis=yaxis,
        xtickfont=font(fontsize), 
        ytickfont=font(fontsize), 
        guidefont=font(fontsize), 
        legendfont=font(fontsize),
        titlefont=font(fontsize))
    for (stat,label) in zip(stats,labels)
        plot!(p,[(1:interval[label]:length(trace),trace[1:interval[label]:length(trace)]) for trace in stat],alpha=alpha,color=colors[label],label="")
        plot!(p,1:interval[label]:length(mean(stat)),mean(stat)[1:interval[label]:length(mean(stat))],color=colors[label],linestyle=:dash,markershape=markers[label],width=2,label=label)
        # plot!(p,1:interval:length(mean(stat)),mean(stat)[1:interval:length(mean(stat))],color=colors[label],linestyle=:dash,markershape=markers[label],width=2,label=label)
    end
    
    return p
end

function plottimetraces(stats,times,labels,colors,markers,interval,alpha,figsize,fontsize,title::String,xlabel::String,legend=:outerright,yaxis=:linear)

    p = plot(size=figsize,legend=legend,title=title,xlabel=xlabel,yaxis=yaxis,
        xtickfont=font(fontsize), 
        ytickfont=font(fontsize), 
        guidefont=font(fontsize), 
        legendfont=font(fontsize),
        titlefont=font(fontsize))
    for (stat,time,label) in zip(stats,times,labels)
        medianTime = median(cumsum(hcat(time...),dims=1),dims=2)[:]
        medianTrace = median(hcat(stat...),dims=2)[:]
        
        plot!(p,[(cumsum(_time),trace) for (_time,trace) in zip(time,stat)],alpha=alpha,color=colors[label],label="")
        plot!(p,(medianTime[1:interval[label]:length(medianTime)],medianTrace[1:interval[label]:length(medianTime)]),color=colors[label],linestyle=:dash,markershape=markers[label],width=2,label=label)
    end
    
    
    return p
end
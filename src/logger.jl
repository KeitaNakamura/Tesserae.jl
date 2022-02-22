import Dates
import ProgressMeter
import ProgressMeter: Progress, BarGlyphs

const PROGRESS_METER_MAX = 10000

mutable struct Logger
    # log
    logpoints::Vector{Float64}
    i::Int
    islogpoint::Bool
    # progress
    prog::Progress
    showprogress::Bool
    color::Symbol
end

function Logger(start::Real, stop::Real, step::Real; showprogress::Bool=false, showspeed::Bool=true, color::Symbol=:yellow)
    @assert start < stop
    logpoints = collect(start:step:stop)
    last(logpoints) < stop && push!(logpoints, stop)
    prog = Progress(
        PROGRESS_METER_MAX;
        barglyphs = BarGlyphs('|', '█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'], ' ', '|'),
        barlen = 20,
        showspeed,
        color,
    )
    Logger(logpoints, -1, false, prog, showprogress, color)
end

t_start(log::Logger) = first(logpoints(log))
t_stop(log::Logger) = last(logpoints(log))

logpoints(logger::Logger) = logger.logpoints
logindex(logger::Logger) = logger.i
islogpoint(logger::Logger) = logger.islogpoint

function isfinised(logger::Logger, t::Real)
    progress_int(logger, t) ≥ PROGRESS_METER_MAX
end

function progress_int(logger::Logger, t::Real)
    t0 = t_start(logger)
    t1 = t_stop(logger)
    floor(Int, PROGRESS_METER_MAX * ((t - t0) / (t1 - t0)))
end

function update!(logger::Logger, t::Real)
    if logger.showprogress
        if logindex(logger) == -1 # time stamp for start
            printstyled("Start: ", Dates.now(); logger.color)
            println()
        end
        int = progress_int(logger, t)
        if int >= PROGRESS_METER_MAX
            ProgressMeter.finish!(logger.prog)
        else
            ProgressMeter.update!(logger.prog, int)
        end
    end
    i = searchsortedlast(logpoints(logger), t) - 1
    if logger.i < i # not yet logged
        logger.i = i
        logger.islogpoint = true
    else
        logger.islogpoint = false
    end
end

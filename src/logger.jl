import Dates
import ProgressMeter

const PROGRESS_METER_MAX = 10000


mutable struct Logger{V <: AbstractVector, P}
    # log
    logpoints::V
    i::Int
    islogpoint::Bool
    # progress
    show_progress::Bool
    pmeter::P
end

function Logger(logpoints::AbstractVector; show_progress::Bool = false)
    @assert issorted(logpoints)
    pmeter = ProgressMeter.Progress(
        PROGRESS_METER_MAX,
        barglyphs = ProgressMeter.BarGlyphs('|','█', ['▌'],' ','|',),
        barlen = 20,
        color = :yellow,
    )
    printstyled("Start: ", Dates.now(); color = :yellow)
    println()
    Logger(logpoints, -1, false, show_progress, pmeter)
end

Base.first(log::Logger) = first(logpoints(log))
Base.last(log::Logger) = last(logpoints(log))

logpoints(logger::Logger) = logger.logpoints
logindex(logger::Logger) = logger.i

function isfinised(logger::Logger, t::Real)
    getprogress(logger, t) ≥ PROGRESS_METER_MAX
end

islogpoint(logger) = logger.islogpoint

function update!(logger::Logger, t::Real)
    logger.show_progress && printprogress(logger, t)
    i = searchsortedlast(logpoints(logger), t) - 1
    if logger.i < i # not yet logged
        logger.i = i
        logger.islogpoint = true
    else
        logger.islogpoint = false
    end
end

function getprogress(logger::Logger, t::Real)
    t0 = first(logger)
    t1 = last(logger)
    floor(Int, PROGRESS_METER_MAX * ((t - t0) / (t1 - t0)))
end

function printprogress(logger::Logger, t::Real)
    perc = getprogress(logger, t)
    if perc >= PROGRESS_METER_MAX
        ProgressMeter.finish!(logger.pmeter)
    else
        ProgressMeter.update!(logger.pmeter, perc)
    end
end

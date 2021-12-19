import Dates
import ProgressMeter

const PROGRESS_METER_MAX = 10000

"""
    Logger(logpoints::AbstractVector; show_progress = false)

Construct logger which handle with time event in the calculation.
The workflow using `Logger` can be written as follows:

```jldoctest
function workflow()
    logpoints = 0:0.5:5
    logger = Logger(logpoints)
    t = 0.0
    dt = 0.2
    timestamps = Float64[]

    while !isfinised(logger, t)
        #
        # 1. Calculations...
        #

        # 2. Update time step and logger
        update!(logger, t += dt)

        #
        # 3. Save data at log point
        #
        if islogpoint(logger)

            # linear index is available for numbering data
            i = logindex(logger)

            push!(timestamps, t)
        end
    end

    timestamps
end

workflow()

# output

11-element Vector{Float64}:
 0.2
 0.6000000000000001
 1.0
 1.5999999999999999
 2.1999999999999997
 2.6
 3.0000000000000004
 3.600000000000001
 4.000000000000001
 4.600000000000001
 5.000000000000002
```

As shown above example, `islogpoint(logger)` is `true` when
time `t` become first greater than or equalt to `logpoints`.
"""
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

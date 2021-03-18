module Loggers

import ProgressMeter

export Logger, islogpoint, logindex, update!, isfinised

"""
    Logger(logpoints::AbstractVector; progress = false)

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

11-element Array{Float64,1}:
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
    progress::Bool
    pmeter::P
end

function Logger(logpoints::AbstractVector; progress::Bool = false)
    @assert issorted(logpoints)
    pmeter = ProgressMeter.Progress(
        10000,
        barglyphs = ProgressMeter.BarGlyphs('|','█', ['▌'],' ','|',),
        barlen = 20,
        color = :yellow,
    )
    Logger(logpoints, 0, false, progress, pmeter)
end

Base.first(log::Logger) = first(logpoints(log))
Base.last(log::Logger) = last(logpoints(log))

logpoints(logger::Logger) = logger.logpoints
logindex(logger::Logger) = logger.i

function isfinised(logger::Logger, t::Real)
    getprogress(logger, t) ≥ 10000
end

islogpoint(logger) = logger.islogpoint

function update!(logger::Logger, t::Real)
    logger.progress && printprogress(logger, t)
    i = searchsortedlast(logpoints(logger), t)
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
    floor(Int, 10000 * (t - t0) / (t1 - t0))
end

function printprogress(logger::Logger, t::Real)
    ProgressMeter.update!(logger.pmeter, getprogress(logger, t))
end

end

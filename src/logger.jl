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
    tlast::Float64
    # print
    nlines::Int
end

function Logger(start::Real, stop::Real, step::Real; showprogress::Bool=false, color::Symbol=:yellow)
    @assert start < stop
    logpoints = collect(start:step:stop)
    last(logpoints) < stop && push!(logpoints, stop)
    prog = Progress(
        PROGRESS_METER_MAX;
        barglyphs = BarGlyphs('|', '█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'], ' ', '|'),
        barlen = 20,
        color,
    )
    Logger(logpoints, -1, false, prog, showprogress, color, time(), 0)
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

function update!(logger::Logger, t::Real; print = nothing)
    if logger.showprogress
        if logindex(logger) == -1 # time stamp for start
            printstyled("Start: ", Dates.now(); logger.color)
            println()
        end

        ansi_moveup(n::Int) = string("\e[", n, "A")
        ansi_movecol1 = "\e[1G"
        ansi_cleartoend = "\e[0J"

        int = progress_int(logger, t)
        isdone = int >= PROGRESS_METER_MAX

        # use own threshold for printing to match timing of printing progress bar and given `print`
        T = time()
        if T > logger.tlast + logger.prog.dt || isdone
            if logger.nlines > 0
                Base.print(ansi_moveup(logger.nlines), ansi_movecol1, ansi_cleartoend)
            end

            if isdone
                ProgressMeter.finish!(logger.prog)
            else
                ProgressMeter.update!(logger.prog, int)
            end

            if print !== nothing
                str = sprint() do iostr
                    !isdone && println(iostr)
                    show(iostr, "text/plain", print)
                end
                logger.nlines = count("\n", str)
                Base.print(str)
                isdone && println()
            end

            # Compensate for any overhead of printing (see ProgressMeter.jl).
            logger.tlast = T + 2*(time()-T)
        end
    end

    i = searchsortedlast(logpoints(logger), t) - 1
    if logger.i < i # not yet logged
        logger.i = i
        logger.islogpoint = true
    else
        logger.islogpoint = false
    end

    nothing
end

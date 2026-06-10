module Progress

using Printf
import Preferences
import ..Tesserae: commas

const ENABLED = Preferences.@load_preference("enable_showprogress_macro", true)

const BAR_FRONTS = ('▏', '▎', '▍', '▌', '▋', '▊', '▉')
const LINE_OVERHEAD = 29

# P2 median estimator. It keeps five moving markers instead of storing all step times.
mutable struct P2Median
    count::Int          # Number of accepted samples.
    q::Vector{Float64}  # Initial samples, then marker heights after five samples.
    n::Vector{Int}      # Current integer marker positions.
    np::Vector{Float64} # Desired marker positions.
    dn::Vector{Float64} # Desired marker increments for min, q1, median, q3, max.
end

P2Median() = P2Median(0, Float64[], zeros(Int, 5), zeros(5), [0.0, 0.25, 0.5, 0.75, 1.0])

# Convert the first five exact samples into the five P2 marker heights and positions.
function initialize_markers!(state::P2Median)
    sort!(state.q)
    state.n .= 1:5
    state.np .= 1:5
    state
end

# Try the parabolic P2 marker update first; callers fall back to linear interpolation
# when the parabolic estimate would break marker ordering.
function parabolic_marker(q, n, i, d)
    left = n[i] - n[i - 1]
    right = n[i + 1] - n[i]
    span = n[i + 1] - n[i - 1]
    q[i] + d / span * ((left + d) * (q[i + 1] - q[i]) / right + (right - d) * (q[i] - q[i - 1]) / left)
end

linear_marker(q, n, i, d) = q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])

# Return the marker interval that contains value, updating the outer markers when
# value becomes the new observed minimum or maximum.
function marker_interval!(q, value)
    value < q[1] && (q[1] = value; return 1)
    value < q[2] && return 1
    value < q[3] && return 2
    value < q[4] && return 3
    value ≤ q[5] && return 4
    q[5] = value
    4
end

function Base.push!(state::P2Median, value)
    isfinite(value) && value ≥ 0 || return state
    state.count += 1
    if state.count ≤ 5
        push!(state.q, value)
        state.count == 5 && initialize_markers!(state)
        return state
    end
    q, n, np = state.q, state.n, state.np
    k = marker_interval!(q, value)
    for i in (k + 1):5
        n[i] += 1
    end
    for i in 1:5
        np[i] += state.dn[i]
    end
    for i in 2:4
        delta = np[i] - n[i]
        if (delta ≥ 1 && n[i + 1] - n[i] > 1) || (delta ≤ -1 && n[i - 1] - n[i] < -1)
            d = delta ≥ 0 ? 1 : -1
            q_next = parabolic_marker(q, n, i, d)
            q[i] = q[i - 1] < q_next < q[i + 1] ? q_next : linear_marker(q, n, i, d)
            n[i] += d
        end
    end
    state
end

function median(state::P2Median)
    state.count == 0 && return Inf
    if state.count < 5
        sorted = sort(state.q)
        mid = state.count ÷ 2
        return isodd(state.count) ? sorted[mid + 1] : (sorted[mid] + sorted[mid + 1]) / 2
    end
    state.q[3]
end

# Meter stores both simulation progress and terminal rendering state so resize-safe
# refreshes can overwrite the previous multi-line display.
mutable struct Meter
    # Simulation progress
    t_start::Float64
    t_stop::Float64
    t_last_print::Float64
    t_prev::Float64
    timestep::Float64
    has_timestep::Bool
    progress::Float64
    count::Int
    # Wall-clock timing and ETA
    wall_init::Float64
    wall_prev::Float64
    wall_last_print::Float64
    rate_ema::Float64
    has_rate::Bool
    eta::Float64
    has_eta::Bool
    alpha::Float64
    # Terminal rendering
    printed::Bool
    done::Bool
    numprintedvalues::Int
    desc::String
    dt::Float64
    output::IO
    color::Symbol
    valuecolor::Symbol
    summary::Bool
    # Step statistics
    step_count::Int
    step_min::Float64
    step_max::Float64
    step_mean::Float64
    step_m2::Float64
    timestep_median::P2Median
    step_median::P2Median
end

function Meter(t_start, t_stop; desc = "Progress: ", dt = 0.1, output = stderr,
               color = :green, valuecolor = :blue, alpha = 0.1, summary = true)
    now = time()
    t_start_float = Float64(t_start)
    t_stop_float = Float64(t_stop)
    Meter(t_start_float, t_stop_float, t_start_float, t_start_float, 0.0, false,
          0.0, 0, now, now, now, 0.0, false, 0.0, false, Float64(alpha), false,
          false, 0, String(desc), Float64(dt), output, color, valuecolor,
          Bool(summary), 0, Inf, -Inf, 0.0, 0.0, P2Median(), P2Median())
end

function durationstring(seconds)
    total_seconds = max(0.0, Float64(seconds))
    days = floor(Int, total_seconds / (60 * 60 * 24))
    rest = total_seconds - 60 * 60 * 24 * days
    hours = floor(Int, rest / (60 * 60))
    rest -= 60 * 60 * hours
    minutes = floor(Int, rest / 60)
    seconds = floor(Int, rest - 60 * minutes)
    hhmmss = @sprintf "%u:%02u:%02u" hours minutes seconds
    days > 9 && return @sprintf "%.2f days" total_seconds / (60 * 60 * 24)
    days > 0 && return @sprintf "%u days, %s" days hhmmss
    hhmmss
end

function timestring(seconds)
    seconds == Inf && return "  N/A  s"
    isfinite(seconds) || return "  N/A  s"
    ns = 1_000_000_000 * seconds
    units = (
        (1, "ns"),
        (1_000, "μs"),
        (1_000_000, "ms"),
        (1_000_000_000, "s"),
        (60 * 1_000_000_000, "m"),
        (60 * 60 * 1_000_000_000, "hr"),
        (24 * 60 * 60 * 1_000_000_000, "d"),
    )
    for (scale, unit) in units
        value = ns / scale
        round(value) < 100 && return @sprintf "%5.2f %2s" value unit
    end
    " >100  d"
end

speedstring(seconds_per_iteration) = timestring(seconds_per_iteration) * "/it"

# Record one wall-clock loop step for the final summary.
function record_step!(p::Meter, seconds)
    isfinite(seconds) && seconds ≥ 0 || return p
    push!(p.step_median, seconds)
    p.step_count += 1
    p.step_min = min(p.step_min, seconds)
    p.step_max = max(p.step_max, seconds)
    # Welford update for summary variance without storing every step time.
    δ = seconds - p.step_mean
    p.step_mean += δ / p.step_count
    p.step_m2 += δ * (seconds - p.step_mean)
    p
end

# Record the simulation-time increment between successive loop updates.
function record_timestep!(p::Meter, seconds)
    isfinite(seconds) && seconds ≥ 0 || return p
    push!(p.timestep_median, seconds)
    p.timestep = seconds
    p.has_timestep = true
    p
end

function barstring(barlen::Integer, percentage_complete)
    barlen ≤ 0 && return ""
    percentage_complete ≥ 100 && return string('|', repeat("█", barlen), '|')
    nbars = barlen * percentage_complete / 100
    nsolid = trunc(Int, nbars)
    frac = nbars - nsolid
    front_idx = round(Int, frac * (length(BAR_FRONTS) + 1))
    front = front_idx > length(BAR_FRONTS) ? '█' :
            front_idx == 0 ? ' ' :
            BAR_FRONTS[front_idx]
    nempty = barlen - nsolid - 1
    string('|', repeat("█", max(0, nsolid)), front, repeat(" ", max(0, nempty)), '|')
end

# Leave room for the description, percentage, and ETA/elapsed text around the bar.
tty_width(p::Meter) = max(0, displaysize(p.output)[2] - length(p.desc) - LINE_OVERHEAD)

# Convert the current simulation time into a monotone 0..1 progress fraction.
function fraction(p::Meter, t, t_stop)
    current = Float64(t)
    stop = Float64(t_stop)
    p.t_stop = stop
    span = stop - p.t_start
    if span ≤ 0
        return ifelse(current ≥ stop, 1.0, 0.0)
    end
    clamp((current - p.t_start) / span, 0.0, 1.0)
end

# Update ETA from the smoothed wall-time-per-simulation-time estimate.
function update_eta!(p::Meter, t, interval_seconds)
    update_rate!(p, t, interval_seconds)
    p.has_rate || return p
    remaining = max(0.0, p.t_stop - Float64(t))
    eta = remaining * p.rate_ema
    isfinite(eta) || return p
    p.eta = eta
    p.has_eta = true
    p
end

function eta(p::Meter)
    p.has_eta || return "N/A"
    durationstring(round(Int, p.eta))
end

# Show current and median simulation timestep in the same width as stepstring.
function timestepstring(p::Meter)
    p.has_timestep || return "N/A"
    current = timestring(p.timestep)
    p.count > 0 || return current
    m = median(p.timestep_median)
    isfinite(m) && m ≥ 0 || return current
    string(current, " (median ", timestring(m), ")")
end

# Show current and median wall-clock time per loop step.
function stepstring(p::Meter, step_seconds)
    current = timestring(step_seconds)
    p.step_count > 0 || return current
    m = median(p.step_median)
    isfinite(m) && m ≥ 0 || return current
    string(current, " (median ", timestring(m), ")")
end

# Build the single-line progress meter; value rows are rendered separately below it.
function line(p::Meter, percentage_complete; finished = false)
    bar = barstring(tty_width(p), percentage_complete)
    spacer = endswith(p.desc, " ") ? "" : " "
    percentage = finished ? 100 : min(99, round(Int, percentage_complete))
    if finished
        elapsed = durationstring(time() - p.wall_init)
        @sprintf "%s%s%3u%%%s Time: %s" p.desc spacer percentage bar elapsed
    else
        @sprintf "%s%s%3u%%%s  ETA: %s" p.desc spacer percentage bar eta(p)
    end
end

function printover(io::IO, msg::AbstractString, color::Symbol)
    print(io, "\r")
    printstyled(io, msg; color)
    # Remove leftovers when the new line is shorter than the previous one.
    print(io, "\e[K")
end

function printstyledln(io::IO, color::Symbol, args...)
    printstyled(io, args...; color)
    println(io)
end

# Clear previously printed value rows before redrawing them at the new terminal width.
function clear_values(io::IO, numlines::Integer)
    for _ in 1:numlines
        # Clear the current value row, then move up toward the progress line.
        print(io, "\r\e[K\e[A")
    end
end

# Print aligned value rows and remember how many terminal rows they occupied.
function printvalues!(p::Meter, showvalues)
    isempty(showvalues) && (p.numprintedvalues = 0; return p)
    maxwidth = maximum(length(string(name)) for (name, _) in showvalues)
    max_len = max(1, displaysize(p.output)[2])
    p.numprintedvalues = 0
    for (name, value) in showvalues
        msg = "\n  " * lpad(string(name) * ": ", maxwidth + 3) * string(value)
        printover(p.output, msg, p.valuecolor)
        # Wrapped value lines need matching cursor-up moves on the next refresh.
        p.numprintedvalues += max(1, ceil(Int, (length(msg) - 1) / max_len))
    end
    p
end

# Redraw the progress line and value rows in-place.
function refresh!(p::Meter, line::AbstractString, showvalues)
    # The cursor is left on the progress line; move to the last value row first.
    print(p.output, "\n" ^ p.numprintedvalues)
    clear_values(p.output, p.numprintedvalues)
    printover(p.output, line, p.color)
    printvalues!(p, showvalues)
    # Leave the cursor on the progress line for the next in-place refresh.
    print(p.output, "\r\e[A" ^ p.numprintedvalues)
    flush(p.output)
    p.printed = true
    p
end

# Smooth wall seconds per simulation-time unit using the latest printed interval.
function update_rate!(p::Meter, t, wall_interval)
    current = Float64(t)
    sim_interval = current - p.t_last_print
    if sim_interval > 0 && isfinite(sim_interval) && wall_interval ≥ 0 && isfinite(wall_interval)
        # ETA is estimated as remaining simulation time times wall time per simulation time.
        rate = wall_interval / sim_interval
        observed = current - p.t_start
        # Give each printed interval at least its elapsed share of the run so early
        # estimates settle faster than a fixed-alpha EMA.
        alpha = p.has_rate && observed > 0 ? max(p.alpha, sim_interval / observed) : 1.0
        p.rate_ema = alpha * rate + (1 - alpha) * p.rate_ema
        p.has_rate = true
    end
    p.t_last_print = current
    p
end

function showvalues(p::Meter, elapsed, step_seconds)
    [(:Elapsed, durationstring(elapsed)),
     (:Iterations, commas(p.count)),
     (:Timestep, timestepstring(p)),
     ("Wall time/step", stepstring(p, step_seconds))]
end

# Called after each loop body execution.
function update!(p::Meter, t, t_stop = p.t_stop)
    p.done && return p
    now = time()
    # Measure only the user's loop body since wall_prev is reset after rendering.
    step_seconds = now - p.wall_prev
    record_step!(p, step_seconds)
    current = Float64(t)
    timestep = current - p.t_prev
    record_timestep!(p, timestep)
    p.t_prev = current
    p.count += 1
    # Keep progress monotone even if the loop variable moves backward slightly.
    p.progress = max(p.progress, fraction(p, t, t_stop))

    if p.progress ≥ 1
        # Do not include final rendering in the last recorded step time.
        p.wall_prev = time()
        return finish!(p)
    end

    if now ≥ p.wall_last_print + p.dt
        interval_seconds = now - p.wall_last_print
        update_eta!(p, t, interval_seconds)
        elapsed = now - p.wall_init
        line_text = line(p, 100 * p.progress)
        refresh!(p, line_text, showvalues(p, elapsed, step_seconds))
        p.wall_last_print = now
    end
    # Reset after refresh so rendering overhead does not inflate the next step.
    p.wall_prev = time()
    p
end

# Clear the live meter, print the completed line, and optionally print the summary.
function finish!(p::Meter)
    p.done && return p
    p.done = true
    # If the loop finished before the first refresh, keep output silent.
    p.printed || return p
    # Move to the bottom of the live display before clearing it upwards.
    print(p.output, "\n" ^ p.numprintedvalues)
    clear_values(p.output, p.numprintedvalues)
    printover(p.output, line(p, 100.0; finished = true), p.color)
    printvalues!(p, [(:Iterations, commas(p.count))])
    println(p.output)
    p.summary && printsummary!(p)
    flush(p.output)
    p.numprintedvalues = 0
    p
end

function step_std(p::Meter)
    p.step_count ≤ 1 && return 0.0
    sqrt(p.step_m2 / (p.step_count - 1))
end

# Print final wall-clock step statistics gathered online during update!.
function printsummary!(p::Meter)
    p.step_count == 0 && return p
    σ = step_std(p)
    printstyledln(p.output, p.valuecolor, "Wall time/step:")
    printstyledln(p.output, p.valuecolor, "  Range (min … max): ", lstrip(timestring(p.step_min)), " … ", lstrip(timestring(p.step_max)))
    printstyledln(p.output, p.valuecolor, "  Mean ± σ:          ", lstrip(timestring(p.step_mean)), " ± ", lstrip(timestring(σ)))
    p
end

"""
```
@showprogress while t < t_stop
    # computation...
end

@showprogress dt=0.5 while t < t_stop
    # computation...
end
```

displays progress of `while` loop.
"""
macro showprogress(args...)
    isempty(args) && throw(ArgumentError("@showprogress requires a while loop"))
    expr = args[end]
    ENABLED || return esc(expr)
    kwargs = map(args[1:end-1]) do arg
        Meta.isexpr(arg, :(=)) ||
            throw(ArgumentError("@showprogress options must be keyword assignments"))
        Expr(:kw, arg.args[1], esc(arg.args[2]))
    end
    Meta.isexpr(expr, :while) || throw(ArgumentError("@showprogress supports only while loops"))
    cnd, blk = expr.args[1], expr.args[2]
    Meta.isexpr(cnd, :call) && cnd.args[1] == :< ||
        throw(ArgumentError("@showprogress supports only `while t < t_stop` loops"))
    t, t_stop = cnd.args[2], cnd.args[3]
    prog = gensym(:meter)
    completed = gensym(:completed)
    quote
        # Macro hygiene resolves these unqualified names to Tesserae.Progress.
        local $prog = Meter($(esc(t)), $(esc(t_stop)); $(kwargs...))
        local $completed = false
        try
            while $(esc(cnd))
                $(esc(blk))
                update!($prog, $(esc(t)), $(esc(t_stop)))
            end
            $completed = true
        finally
            $completed && finish!($prog)
        end
    end
end

end

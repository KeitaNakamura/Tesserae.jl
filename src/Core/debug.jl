# -----------------------------------------------------------------------------
#  Debug mode
# -----------------------------------------------------------------------------

const DEBUG = Preferences.@load_preference("debug_mode", false)

@static if DEBUG
    @eval macro debug(ex)
        return :($(esc(ex)))
    end
else
    @eval macro debug(ex)
         return nothing
    end
end

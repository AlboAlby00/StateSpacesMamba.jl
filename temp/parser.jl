using ArgParse

function parse_command_line_args()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--scan"
            help = "Select 'cumsum' or 'logcumsumexp'"
            arg_type = String
            required = true
            default = "cumsum"
    end

    return parse_args(ARGS, s)
end
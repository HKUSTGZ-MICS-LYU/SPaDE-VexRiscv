
set outputDir {REPORT_PATH}
file mkdir $outputDir

read_verilog [ glob {RTL} ]

set_host_options -max_cores 8

set_app_var target_library {PDK}

set_app_var link_library {PDK}

current_design {TOP_NAME}

create_clock -period {CLOCK_PERIOD} [get_ports {CLOCK_NAME}] 

compile

report_qor > {REPORT_PATH}/dc_qor.rpt
report_power > {REPORT_PATH}/dc_power.rpt 

exit
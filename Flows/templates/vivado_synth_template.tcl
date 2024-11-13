# STEP#1: define the output directory area.
#
set outputDir {REPORT_PATH}
file mkdir $outputDir
#
# STEP#2: setup design sources and constraints
#
read_verilog [ glob {RTL} ]
#
# STEP#3: run synthesis, write design checkpoint, report timing, 
# and utilization estimates
#
synth_design -top {TOP_NAME} -part {FPGA_TARGET}

# STEP#4: run logic optimization, placement and physical logic optimization, 
# write design checkpoint, report utilization and timing estimates
#

create_clock -period {CLOCK_PERIOD} [get_ports {CLOCK_NAME}] 

opt_design
place_design
phys_opt_design
report_utilization -file $outputDir/util.rpt
#
# STEP#5: run the router, write the post-route design checkpoint, report the routing
# status, report timing, power, and DRC, and finally save the Verilog netlist.
#
route_design
report_timing_summary -file $outputDir/timing_summary.rpt
report_power -file $outputDir/power.rpt
report_drc -file $outputDir/drc.rpt
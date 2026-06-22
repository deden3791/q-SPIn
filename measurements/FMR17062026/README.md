## Set up 

Date measurements taken: 17/06/2026

Set up: smiley face microstrip with magnetic samples in external magnetic field

vna.set_bandwidth(500)
vna.set_power(-20)
vna.set_startfreq(1e9) 
vna.set_stopfreq(18e9)
vna.set_nop(500)
vna.set_averages(1)
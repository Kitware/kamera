sudo watch -n5 '(chronyc sources ; echo "" ; ntpshmmon -t 1 ; echo "" ; phc_ctl enp1s0 get cmp)'

# Allows you to change the auto exposure behavior of the cameras to either under or
# over exposure the imagery. A negative value under exposes the imagery, a postive
# value overexposes.
# Appropriate range seems to be in the single digits.
EXP_PARAM=0

docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas0/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Exposure Comp.=${EXP_PARAM}'\""
docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas1/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Exposure Comp.=${EXP_PARAM}'\""
docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas2/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Exposure Comp.=${EXP_PARAM}'\""

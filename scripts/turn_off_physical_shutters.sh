# Sets shutter to ES mode

docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas0/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Shutter Mode=2'\""
docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas1/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Shutter Mode=2'\""
docker exec -it cam-rgb-center bash -c "source /root/kamera/activate_ros.bash && rosservice call /cas2/rgb/rgb_driver/set_phaseone_parameter \"parameters: 'Shutter Mode=2'\""

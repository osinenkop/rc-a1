# rc-a1
Unitree A1 simulation and control

## Description
This is a controller subroutine for running MPC and RQL.
## Requirements
The package is supposed to be run with the special [QuadSDK branch] (just install in our own docker image).
## Usage
 - Launch the simulation with:
```
roslaunch quad_utils quad_gazebo.launch
```
 - Stand the robot with:
```
rostopic pub /robot_1/control/mode std_msgs/UInt8 "data: 1"
```
 - Run MPC or RQL:
```
rosrun rql_mpc main.py
```
 - Run the stack with twist input:
```
roslaunch quad_utils quad_plan_calf.launch reference:=twist 
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/robot_1/cmd_vel
```

## MPC, RQL Tuning:

### To run MPC
In main.py

Uncomment ActorMPCA1 and past in controller instead of ActorRQLA1.
dont forget to set is_fixed_critic_weights=True. (The critic model formally in the controller but not utilised by MPC, so critic weights update is not necessary)

Tune parameters:

- "QandR": - objective weights
- "final_time": - simulation end time
- "prediction_horizon": - N

### To run RQL
In main.py

Uncomment ActorRQLA1 and past in controller instead of ActorMPCA1.
dont forget to set is_fixed_critic_weights=False.



Tune parameters:

- "QandR": - objective weights
- "final_time": - simulation end time
- "prediction_horizon": - N

In critic model.
- single_weight_min= min model weights.
- single_weight_max= max model weights.
In critic
- data_buffer_size= experience replay
- batch_size= batch sizes taken in experience replay 
- td_n= number of temporal diffs if critic update
- critic_regularization_param= number that force remain weights the same (the higher number makes model more conservative)




[QuadSDK branch]: https://github.com/Slavoch/Legged_robotics/tree/calf
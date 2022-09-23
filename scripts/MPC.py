#!/usr/bin/env python
import numpy as np
import casadi as ca     # casadi求解器，下面MPC的非线性优化问题将使用casadi求解
import time

def MPC(self_state, goal_state):
    opti = ca.Opti()
    ## parameters for optimization
    T = 0.020                           # MPC的离散时间，需要与local_planner.py的self.replan_period相等
    N = 50                              # MPC的预测步长，需要与local_planner.py的self.N保持相等
    v_max = 1.5                         # 约束的小车最大线速度
    omega_max = 1.0                     # 约束的小车最大角速度
    Q = np.array([[2.0, 0.0, 0.0],[0.0, 2.0, 0.0],[0.0, 0.0, 1.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.4]])
    goal = goal_state[:,:3]
    opt_x0 = opti.parameter(3)
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]

    ## state variables
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    ## create function for F(x)

    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]), u_[0]*ca.sin(x_[2]), u_[1]])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)

    # Admissable Control constraints
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    # System Model constraints
    for i in range(N):
        x_next = opt_states[i, :] + T*f(opt_states[i, :], opt_controls[i, :]).T
        opti.subject_to(opt_states[i+1, :]==x_next)

    #### cost function
    obj = 0 
    for i in range(N):
        obj = obj + 0.8*ca.mtimes([(opt_states[i, :] - goal[[i]]), Q, (opt_states[i, :]- goal[[i]]).T]) + 0.5*ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T]) 
    obj = obj + 2*ca.mtimes([(opt_states[0, :] - goal[[0]]), Q, (opt_states[0, :]- goal[[0]]).T])

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-4, 'ipopt.acceptable_obj_change_tol':1e-4}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, self_state[:,:3])

    try:
        sol = opti.solve()                  # 优化!!!!!!!!!!!!!
        u_res = sol.value(opt_controls)     # 获得最优控制输入序列
        state_res = sol.value(opt_states)   # 获得最优状态序列
    except:
        state_res = np.repeat(self_state[:3],N+1,axis=0)
        u_res = np.zeros([N,2])

    return state_res, u_res

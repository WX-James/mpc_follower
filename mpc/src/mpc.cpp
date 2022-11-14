#include "mpc.h"
 
using namespace std;

void MPC::init(ros::NodeHandle &nh)
{
    nh.param("mpc/du_threshold", du_th, -1.0);
    nh.param("mpc/dt", dt, -1.0);
    nh.param("mpc/max_iter", max_iter, -1);
    nh.param("mpc/predict_steps", T, -1);
    nh.param("mpc/delay_num", delay_num, -1);
    nh.param("mpc/max_omega", max_omega, -1.0);
    nh.param("mpc/max_domega", max_domega, -1.0);
    nh.param("mpc/max_speed", max_speed, -1.0);
    nh.param("mpc/max_accel", max_accel, -1.0);
    nh.param("mpc/tolerance", tolerance, 0.1);
    nh.param("mpc/control_a", control_a, false);
    nh.param<std::vector<double>>("mpc/matrix_q", Q, std::vector<double>());
    nh.param<std::vector<double>>("mpc/matrix_r", R, std::vector<double>());
    nh.param<std::vector<double>>("mpc/matrix_rd", Rd, std::vector<double>());

    cout<<"du_t="<<du_th<<endl;

    has_odom = false;
    receive_traj_ = false;
    max_comega = max_domega * dt;
    max_cv = max_accel * dt;
    xref = Eigen::Matrix<double, 4, 500>::Zero(4, 500);
    last_output = output = dref = Eigen::Matrix<double, 2, 500>::Zero(2, 500);
    for (int i=0; i<delay_num; i++)
        output_buff.push_back(Eigen::Vector2d::Zero());

    cmd_vel.angular.z = 0.0;
    cmd_vel.linear.x = 0.0;

    has_cloud = false;

    cmd_twist_pub_ = nh.advertise<geometry_msgs::Twist>("cmd_vel_", 200);
    vis_pub = nh.advertise<visualization_msgs::Marker>("/following_path", 10);
    predict_pub = nh.advertise<visualization_msgs::Marker>("/predict_path", 10);
    ref_pub = nh.advertise<visualization_msgs::Marker>("/reference_path", 10);

    cmd_timer_ = nh.createTimer(ros::Duration(0.01), &MPC::cmdCallback, this);
    odom_sub_ = nh.subscribe("odom", 1, &MPC::rcvOdomCallBack, this);
    traj_sub_ = nh.subscribe("traj", 1, &MPC::rcvTrajCallBack, this);
}

void MPC::rcvTrajCallBack(mpc::PolynomeConstPtr msg)
{
    receive_traj_ = true;
    traj_analyzer.setTraj(msg);
}

void MPC::rcvOdomCallBack(nav_msgs::OdometryPtr msg)
{
    has_odom = true;
    now_state.x = msg->pose.pose.position.x;
    now_state.y = msg->pose.pose.position.y;
    now_state.z = msg->pose.pose.position.z;

    Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z);
    Eigen::Matrix3d R(q);
    now_state.theta = atan2(R.col(0)[1],R.col(0)[0]);

    Eigen::Vector2d lvel(msg->twist.twist.linear.x,msg->twist.twist.linear.y);
    now_state.v = lvel.norm();
}

void MPC::cmdCallback(const ros::TimerEvent &e)
{   
    // drawRefPath();
    // drawFollowPath();
    clock_t time_start,time_end;  //定义clock_t变量用于计时
    time_start = clock();          //搜索开始时间

    
    if (!has_odom || !receive_traj_)
        return;

    vector<TrajPoint> P = traj_analyzer.getRefPoints(T, dt);

    if (traj_analyzer.at_goal)
    {
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
    }
    else
    {
        if (control_a)
        {
        
            for (int i=0; i<T; i++)
            {
                xref(0, i) = P[i].x;
                xref(1, i) = P[i].y;
                xref(2, i) = P[i].v;
                xref(3, i) = P[i].theta;
                dref(0, i) = 0.0;
                dref(1, i) = 0.0;
            }
        }
        else
        {
            for (int i=0; i<T; i++)
            {
                xref(0, i) = P[i].x;
                xref(1, i) = P[i].y;
                xref(3, i) = P[i].theta;
                dref(0, i) = 0.0;
                dref(1, i) = 0.0;
            }
        }

        smooth_yaw();
        getCmd();

    }

    time_end = clock();   //搜索结束时间
    double t_search = double(time_end-time_start)/CLOCKS_PER_SEC;
    // std::cout << "MPC cost time := " << t_search << std::endl;
    cmd_twist_pub_.publish(cmd_vel);
    
    // ROS_INFO("in MPC, the cmd is: linear=%f, angular=%f", cmd.speed, cmd.omega);
}

void MPC::getLinearModel(const MPCState& s)
{
    if (control_a)
    {
        A = Eigen::Matrix4d::Identity();
        A(0, 2) = dt * cos(s.theta);
        A(1, 2) = dt * sin(s.theta);
        A(0, 3) = -s.v * A(1, 2);
        A(1, 3) = s.v * A(0, 2);

        B = Eigen::Matrix<double, 4, 2>::Zero();
        B(2, 0) = dt;
        B(3, 1) = dt;

        C = Eigen::Vector4d::Zero();
        C(0) = -A(0, 3) * s.theta; 
        C(1) = -A(1, 3) * s.theta;
    }
    else
    {
        B = Eigen::Matrix<double, 3, 2>::Zero();
        B(0, 0) = cos(s.theta) * dt;
        B(1, 0) = sin(s.theta) * dt;
        B(2, 1) = dt;

        A = Eigen::Matrix3d::Identity();
        A(0, 2) = -B(1, 0) * s.v;
        A(1, 2) = B(0, 0) * s.v;

        C = Eigen::Vector3d::Zero();
        C(0) = -A(0, 2) * s.theta; 
        C(1) = -A(1, 2) * s.theta; 
    }
}

void MPC::stateTrans(MPCState& s, double a, double yaw_dot)
{
    if (control_a)
    {
        if (yaw_dot >= max_omega)
        {
            yaw_dot = max_omega;
        }else if (yaw_dot<= - max_omega)
        {
            yaw_dot = -max_omega;
        }

        s.x = s.x + s.v * cos(s.theta) * dt;
        s.y = s.y + s.v * sin(s.theta) * dt;
        s.theta = s.theta + yaw_dot * dt;
        s.v = s.v + a * dt;

        if (s.v >= max_speed)
        {
            s.v = max_speed;
        }else if (s.v<= -max_speed)
        {
            s.v = -max_speed;
        }
    }
    else
    {
        if (yaw_dot >= max_omega)
        {
            yaw_dot = max_omega;
        }else if (yaw_dot<= - max_omega)
        {
            yaw_dot = -max_omega;
        }
        if (s.v >= max_speed)
        {
            s.v = max_speed;
        }else if (s.v<= -max_speed)
        {
            s.v = -max_speed;
        }

        s.x = s.x + a * cos(s.theta) * dt;
        s.y = s.y + a * sin(s.theta) * dt;
        s.theta = s.theta + yaw_dot * dt;
        s.v = a;
    }
}

void MPC::predictMotion(void)
{
    xbar[0] = now_state;

    MPCState temp = now_state;
    for (int i=1; i<T+1; i++)
    {
        stateTrans(temp, output(0, i-1), output(1, i-1));
        xbar[i] = temp;
    }
}

void MPC::predictMotion(MPCState* b)
{
    b[0] = xbar[0];

    Eigen::MatrixXd Ax;
    Eigen::MatrixXd Bx;
    Eigen::MatrixXd Cx;
    Eigen::MatrixXd xnext;
    MPCState temp = xbar[0];
    if (control_a)
        for (int i=1; i<T+1; i++)
        {
            Ax = Eigen::Matrix4d::Identity();
            Ax(0, 2) = dt * cos(xbar[i-1].theta);
            Ax(1, 2) = dt * sin(xbar[i-1].theta);
            Ax(0, 3) = -xbar[i-1].v * Ax(1, 2);
            Ax(1, 3) = xbar[i-1].v * Ax(0, 2);
            Ax(3, 2) = 0.0;

            Bx = Eigen::Matrix<double, 4, 2>::Zero();
            Bx(2, 0) = dt;
            Bx(3, 1) = dt;

            Cx = Eigen::Vector4d::Zero();
            Cx(0) = -Ax(0, 3) * xbar[i-1].theta;
            Cx(1) = -Ax(1, 3) * xbar[i-1].theta;
            
            xnext = Ax*Eigen::Vector4d(temp.x, temp.y, temp.v, temp.theta) + Bx*Eigen::Vector2d(output(0, i-1), output(1, i-1)) + Cx;
            temp.x = xnext(0);
            temp.y = xnext(1);
            temp.v = xnext(2);
            temp.theta = xnext(3);
            b[i] = temp;
        }
    else
         
        for (int i=1; i<T+1; i++)
        {  
            Bx = Eigen::Matrix<double, 3, 2>::Zero();
            Bx(0, 0) = cos(xbar[i-1].theta) * dt;
            Bx(1, 0) = sin(xbar[i-1].theta) * dt;
            Bx(2, 1) = dt;

            Ax = Eigen::Matrix3d::Identity();
            Ax(0, 2) = -Bx(1, 0) * xbar[i-1].v;
            Ax(1, 2) = Bx(0, 0) * xbar[i-1].v;

            Cx = Eigen::Vector3d::Zero();
            Cx(0) = -Ax(0, 2) * xbar[i-1].theta; 
            Cx(1) = -Ax(1, 2) * xbar[i-1].theta;
            xnext = Ax*Eigen::Vector3d(temp.x, temp.y, temp.theta) + Bx*Eigen::Vector2d(output(0, i-1), output(1, i-1)) + Cx;
            temp.x = xnext(0);
            temp.y = xnext(1);
            temp.theta = xnext(2);
            b[i] = temp;
        }

}

void MPC::solveMPCA(void)
{
    const int dimx = 4 * (T - delay_num);
    const int dimu = 2 * (T - delay_num);
    const int nx = dimx + dimu;

    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nx);
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;

    // first-order
    for (int i=0, j=delay_num; i<dimx; i+=4, j++)
    {
        gradient[i] = -2 * Q[0] * xref(0, j);
        gradient[i+1] = -2 * Q[1] * xref(1, j);
        gradient[i+2] = -2 * Q[2] * xref(2, j);
        gradient[i+3] = -2 * Q[3] * xref(3, j);
    }

    // second-order
    const int nnzQ = nx + dimu - 2;
    int irowQ[nnzQ];
    int jcolQ[nnzQ];
    double dQ[nnzQ];
    for (int i=0; i<nx; i++)
    {
        irowQ[i] = jcolQ[i] = i;
    }
    for (int i=nx; i<nnzQ; i++)
    {
        irowQ[i] = i - dimu + 2;
        jcolQ[i] = i - dimu;
    }
    for (int i=0; i<dimx; i+=4)
    {
        dQ[i] = Q[0] * 2.0;
        dQ[i+1] = Q[1] * 2.0;
        dQ[i+2] = Q[2] * 2.0;
        dQ[i+3] = Q[3] * 2.0;
    }
    dQ[dimx] = dQ[nx-2] = (R[0] + Rd[0]) * 2.0;
    dQ[dimx + 1] = dQ[nx-1] = (R[1] + Rd[1]) * 2.0;
    for (int i=dimx+2; i<nx-2; i+=2)
    {
        dQ[i] = 2 * (R[0] + 2 * Rd[0]);
        dQ[i+1] = 2 * (R[1] + 2 * Rd[1]);
    }
    for (int i=nx; i<nnzQ; i+=2)
    {
        dQ[i] = -Rd[0] * 2.0;
        dQ[i+1] = -Rd[1] * 2.0;
    }
    hessian.resize(nx, nx);
    Eigen::MatrixXd QQ(nx, nx);
    for (int i=0; i<nx; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
    }
    for (int i=nx; i<nnzQ; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
        hessian.insert(jcolQ[i], irowQ[i]) = dQ[i];
    }

    // equality constraints
    MPCState temp = xbar[delay_num];
    getLinearModel(temp);
    int my = dimx;
    double b[my];
    const int nnzA = 14 * (T-delay_num) - 8;
    int irowA[nnzA];
    int jcolA[nnzA];
    double dA[nnzA];
    Eigen::Vector4d temp_vec(temp.x, temp.y, temp.v, temp.theta);
    Eigen::Vector4d temp_b = A*temp_vec + C;
    
    for (int i=0; i<dimx; i++)
    {
        irowA[i] = jcolA[i] = i;
        dA[i] = 1;
    }
    b[0] = temp_b[0];
    b[1] = temp_b[1];
    b[2] = temp_b[2];
    b[3] = temp_b[3];
    irowA[dimx] = 2;
    jcolA[dimx] = dimx;
    dA[dimx] = -B(2, 0);
    irowA[dimx+1] = 3;
    jcolA[dimx+1] = dimx+1;
    dA[dimx+1] = -B(3, 1);
    int ABidx = 10*(T-delay_num) - 10;
    int ABbegin = dimx+2;
    for (int i=0, j=1; i<ABidx; i+=10, j++)
    {
        getLinearModel(xbar[j+delay_num]);
        for (int k=0; k<4; k++)
        {
            b[4*j+k] = C[k];
            irowA[ABbegin + i + k] = 4*j + k;
            jcolA[ABbegin + i + k] = irowA[ABbegin + i + k] - 4;
            dA[ABbegin + i + k] = -A(k, k);
        }
        irowA[ABbegin + i + 4] = 4*j;
        jcolA[ABbegin + i + 4] = 4*j - 2;
        dA[ABbegin + i + 4] = -A(0, 2);
        
        irowA[ABbegin + i + 5] = 4*j;
        jcolA[ABbegin + i + 5] = 4*j - 1;
        dA[ABbegin + i + 5] = -A(0, 3);

        irowA[ABbegin + i + 6] = 4*j + 1;
        jcolA[ABbegin + i + 6] = 4*j - 2;
        dA[ABbegin + i + 6] = -A(1, 2);

        irowA[ABbegin + i + 7] = 4*j + 1;
        jcolA[ABbegin + i + 7] = 4*j - 1;
        dA[ABbegin + i + 7] = -A(1, 3);
        
        irowA[ABbegin + i + 8] = 4*j +2;
        jcolA[ABbegin + i + 8] = dimx + 2*j;
        dA[ABbegin + i + 8] = -B(2, 0);
        
        irowA[ABbegin + i + 9] = 4*j + 3;
        jcolA[ABbegin + i + 9] = dimx + 2*j + 1;
        dA[ABbegin + i + 9] = -B(3, 1);
    }

    // iequality constraints
    const int mz  = (T-delay_num) - 1;
    const int nnzC = dimu - 2;
    int   irowC[nnzC];
    int   jcolC[nnzC];
    double   dC[nnzC];
    for (int i=0, j=0; i<mz; i++, j+=2)
    {
        irowC[j] = i;
        irowC[j+1] = i;
        jcolC[j] = dimx + 1 + j;
        jcolC[j+1] = jcolC[j] +2;
        dC[j] = -1.0;
        dC[j+1] = 1.0;
    }

    // xlimits and all
    int mx = 3*(T-delay_num);
    int nc = mx+my+mz;
    lowerBound.resize(nc);
    upperBound.resize(nc);
    linearMatrix.resize(nc, nx);
    for (int i=0, j=0, k=0; i<mx; i+=3, j+=4, k+=2)
    {
        lowerBound[i] = -max_speed;
        lowerBound[i+1] = -max_accel;
        lowerBound[i+2] = -max_omega;
        upperBound[i] = max_speed;
        upperBound[i+1] = max_accel;
        upperBound[i+2] = max_omega;
        linearMatrix.insert(i, j+2) = 1;
        linearMatrix.insert(i+1, dimx+k) = 1;
        linearMatrix.insert(i+2, dimx+k+1) = 1;
    }
    for (int i=0; i<nnzA; i++)
    {
        linearMatrix.insert(irowA[i]+mx, jcolA[i]) = dA[i];
    }
    for (int i=0; i<my; i++)
    {
        lowerBound[mx+i] = upperBound[mx+i] = b[i];
    }
    for (int i=0; i<nnzC; i++)
    {
        linearMatrix.insert(irowC[i]+mx+my, jcolC[i]) = dC[i];
    }
    for (int i=0; i<mz; i++)
    {
        lowerBound[mx+my+i] = -max_comega;
        upperBound[mx+my+i] = max_comega;
    }

    // instantiate the solver
    OsqpEigen::Solver solver;

    // settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.settings()->setAbsoluteTolerance(1e-6);
    solver.settings()->setMaxIteration(30000);
    solver.settings()->setRelativeTolerance(1e-6);

    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(nx);
    solver.data()->setNumberOfConstraints(nc);
    if(!solver.data()->setHessianMatrix(hessian)) return;
    if(!solver.data()->setGradient(gradient)) return;
    if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return;
    if(!solver.data()->setLowerBound(lowerBound)) return;
    if(!solver.data()->setUpperBound(upperBound)) return;

    // instantiate the solver
    if(!solver.initSolver()) return;

    // controller input and QPSolution vector
    Eigen::VectorXd QPSolution;

    // solve the QP problem
    if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return;

    // get the controller input
    QPSolution = solver.getSolution();
    for (int i=0; i<delay_num; i++)
    {
        output(0, i) = output_buff[i][0];
        output(1, i) = output_buff[i][1];
    }
    for (int i=0, j=0; i<dimu; i+=2, j++)
    {
        output(0, j+delay_num) = QPSolution[dimx+i];
        output(1, j+delay_num) = QPSolution[dimx+i+1];
    }
}

void MPC::solveMPCV(void)
{
    const int dimx = 3 * (T - delay_num);
    const int dimu = 2 * (T - delay_num);
    const int nx = dimx + dimu;

    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nx);
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;

    // first-order
    for (int i=0, j=delay_num, k=0; i<dimx; i+=3, j++, k+=2)
    {
        gradient[i] = -2 * Q[0] * xref(0, j);
        gradient[i+1] = -2 * Q[1] * xref(1, j);
        gradient[i+2] = -2 * Q[3] * xref(3, j);
        gradient[dimx+k] = -2 * Q[2] * dref(0, j);
    }

    // second-order
    const int nnzQ = nx + dimu - 2;
    int irowQ[nnzQ];
    int jcolQ[nnzQ];
    double dQ[nnzQ];
    for (int i=0; i<nx; i++)
    {
        irowQ[i] = jcolQ[i] = i;
    }
    for (int i=nx; i<nnzQ; i++)
    {
        irowQ[i] = i - dimu + 2;
        jcolQ[i] = i - dimu;
    }
    for (int i=0; i<dimx; i+=3)
    {
        dQ[i] = Q[0] * 2.0;
        dQ[i+1] = Q[1] * 2.0;
        dQ[i+2] = Q[3] * 2.0;
    }
    dQ[dimx] = dQ[nx-2] = (R[0] + Rd[0] + Q[2]) * 2.0;
    dQ[dimx + 1] = dQ[nx-1] = (R[1] + Rd[1]) * 2.0;
    for (int i=dimx+2; i<nx-2; i+=2)
    {
        dQ[i] = 2 * (R[0] + 2 * Rd[0] + Q[2]);
        dQ[i+1] = 2 * (R[1] + 2 * Rd[1]);
    }
    for (int i=nx; i<nnzQ; i+=2)
    {
        dQ[i] = -Rd[0] * 2.0;
        dQ[i+1] = -Rd[1] * 2.0;
    }
    hessian.resize(nx, nx);
    Eigen::MatrixXd QQ(nx, nx);
    for (int i=0; i<nx; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
    }
    for (int i=nx; i<nnzQ; i++)
    {
        hessian.insert(irowQ[i], jcolQ[i]) = dQ[i];
        hessian.insert(jcolQ[i], irowQ[i]) = dQ[i];
    }

    // equality constraints
    MPCState temp = xbar[delay_num];
    getLinearModel(temp);
    int my = dimx;
    double b[my];
    const int nnzA = 11 * (T-delay_num) - 5;
    int irowA[nnzA];
    int jcolA[nnzA];
    double dA[nnzA];
    Eigen::Vector3d temp_vec(temp.x, temp.y, temp.theta);
    Eigen::Vector3d temp_b = A*temp_vec + C;
    
    for (int i=0; i<dimx; i++)
    {
        irowA[i] = jcolA[i] = i;
        dA[i] = 1;
    }
    b[0] = temp_b[0];
    b[1] = temp_b[1];
    b[2] = temp_b[2];
    irowA[dimx] = 0;
    jcolA[dimx] = dimx;
    dA[dimx] = -B(0, 0);
    irowA[dimx+1] = 1;
    jcolA[dimx+1] = dimx;
    dA[dimx+1] = -B(1, 0);
    irowA[dimx+2] = 2;
    jcolA[dimx+2] = dimx+1;
    dA[dimx+2] = -B(2, 1);
    int ABidx = 8*(T-delay_num) - 8;
    int ABbegin = dimx+3;
    for (int i=0, j=1; i<ABidx; i+=8, j++)
    {
        getLinearModel(xbar[j+delay_num]);
        for (int k=0; k<3; k++)
        {
            b[3*j+k] = C[k];
            irowA[ABbegin + i + k] = 3*j + k;
            jcolA[ABbegin + i + k] = irowA[ABbegin + i + k] - 3;
            dA[ABbegin + i + k] = -A(k, k);
        }
        irowA[ABbegin + i + 3] = 3*j;
        jcolA[ABbegin + i + 3] = 3*j - 1;
        dA[ABbegin + i + 3] = -A(0, 2);

        irowA[ABbegin + i + 4] = 3*j + 1;
        jcolA[ABbegin + i + 4] = 3*j - 1;
        dA[ABbegin + i + 4] = -A(1, 2);
        
        irowA[ABbegin + i + 5] = 3*j;
        jcolA[ABbegin + i + 5] = dimx + 2*j;
        dA[ABbegin + i + 5] = -B(0, 0);
        
        irowA[ABbegin + i + 6] = 3*j + 1;
        jcolA[ABbegin + i + 6] = dimx + 2*j;
        dA[ABbegin + i + 6] = -B(1, 0);
        
        irowA[ABbegin + i + 7] = 3*j + 2;
        jcolA[ABbegin + i + 7] = dimx + 2*j + 1;
        dA[ABbegin + i + 7] = -B(2, 1);
    }

    // iequality constraints
    const int mz  = 2 * (T-delay_num) - 2;
    const int nnzC = 2 * dimu - 4;
    int   irowC[nnzC];
    int   jcolC[nnzC];
    double   dC[nnzC];
    for (int i=0, k=0; i<mz; i+=2, k+=4)
    {
        irowC[k] = i;
        jcolC[k] = dimx  + i;
        dC[k] = -1.0;

        irowC[k+1] = i;
        jcolC[k+1] = jcolC[k] +2;
        dC[k+1] = 1.0;

        irowC[k+2] = i + 1;
        jcolC[k+2] = dimx + 1 + i;
        dC[k+2] = -1.0;

        irowC[k+3] = i + 1;
        jcolC[k+3] = jcolC[k+2] +2;
        dC[k+3] = 1.0;
    }

    // xlimits and all
    int mx = dimu;
    int nc = mx+my+mz;
    lowerBound.resize(nc);
    upperBound.resize(nc);
    linearMatrix.resize(nc, nx);
    for (int i=0; i<mx; i+=2)
    {
        lowerBound[i] = -max_speed;
        lowerBound[i+1] = -max_omega;
        upperBound[i] = max_speed;
        upperBound[i+1] = max_omega;
        linearMatrix.insert(i, dimx+i) = 1;
        linearMatrix.insert(i+1, dimx+i+1) = 1;
    }

    for (int i=0; i<nnzA; i++)
    {
        linearMatrix.insert(irowA[i]+mx, jcolA[i]) = dA[i];
    }

    for (int i=0; i<my; i++)
    {
        lowerBound[mx+i] = upperBound[mx+i] = b[i];
    }

    for (int i=0; i<nnzC; i++)
    {
        linearMatrix.insert(irowC[i]+mx+my, jcolC[i]) = dC[i];
    }

    for (int i=0; i<mz; i+=2)
    {
        lowerBound[mx+my+i] = -max_cv;
        upperBound[mx+my+i] = max_cv;
        lowerBound[mx+my+i+1] = -max_comega;
        upperBound[mx+my+i+1] = max_comega;
    }

    // instantiate the solver
    OsqpEigen::Solver solver;

    // settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.settings()->setAbsoluteTolerance(1e-6);
    solver.settings()->setMaxIteration(30000);
    solver.settings()->setRelativeTolerance(1e-6);

    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(nx);
    solver.data()->setNumberOfConstraints(nc);
    if(!solver.data()->setHessianMatrix(hessian)) return;
    if(!solver.data()->setGradient(gradient)) return;
    if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return;
    if(!solver.data()->setLowerBound(lowerBound)) return;
    if(!solver.data()->setUpperBound(upperBound)) return;

    // instantiate the solver
    if(!solver.initSolver()) return;

    // controller input and QPSolution vector
    Eigen::VectorXd QPSolution;

    // solve the QP problem
    if(solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return;

    // get the controller input
    QPSolution = solver.getSolution();
    // ROS_INFO("Solution: v0=%f     omega0=%f", QPSolution[dimx], QPSolution[dimx+1]);
    for (int i=0; i<delay_num; i++)
    {
        output(0, i) = output_buff[i][0];
        output(1, i) = output_buff[i][1];
    }
    for (int i=0, j=0; i<dimu; i+=2, j++)
    {
        output(0, j+delay_num) = QPSolution[dimx+i];
        output(1, j+delay_num) = QPSolution[dimx+i+1];
    }
}

void MPC::getCmd(void)
{
    int iter;
    ros::Time begin = ros::Time::now();
    for (iter=0; iter<max_iter; iter++)
    {
        predictMotion();
        last_output = output;
        if (control_a)
            solveMPCA();
        else
            solveMPCV();
        double du = 0;
        for (int i=0; i<output.cols(); i++)
        {
            du = du + fabs(output(0, i) - last_output(0, i))+ fabs(output(1, i) - last_output(1, i));
        }
        // break;
        if (du <= du_th || (ros::Time::now()-begin).toSec()>0.01)
        {
            break;
        }
    }
    if (iter == max_iter)
    {
        ROS_WARN("MPC Iterative is max iter");
    }

    predictMotion(xopt);

    drawRefPath();

    // drawFollowPath();

    drawPredictPath(xopt);
    if (control_a)
    {
        cmd_vel.linear.x = now_state.v + dt * output(0, delay_num);
    }
    else
    {
        cmd_vel.linear.x = output(0, delay_num);
    }
        
    cmd_vel.angular.z = output(1, delay_num);

    if (delay_num>0)
    {
        output_buff.erase(output_buff.begin());
        output_buff.push_back(Eigen::Vector2d(output(0, delay_num),output(1, delay_num)));
    }
}

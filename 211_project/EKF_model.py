import numpy as np
from numpy.linalg import inv
import math 
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

import localization_utils



from nav_msgs.msg import Odometry


class RobotEKF(Node):

     def __init__(self,dim_x=3,dim_u=2,eval_mu=None,eval_Gt=None,eval_Vt=None):

        super().__init__('EKF_prediction')

        self.coordinates_x=self.declare_parameter('x',[1.207,2.975,4.074,3.744,0.922,2.563,3.951,2.209,4.556,4.314]).value
        self.coordinates_y=self.declare_parameter('y',0.769,1.358,0.579,0.309,-0.746,0.421,-0.515,0.710,1.251,-0.452).value
        
        self.id=self.declare_parameter('id', [1,2,3,4,5,6,7,8,9,10,11,12,13,14]).value
        
        

        self.alpha_1=self.declare_parameter('alpha_1',0.1).value
        self.alpha_2=self.declare_parameter('alpha_2',0.1).value
        self.alpha_3=self.declare_parameter('alpha_3',0.1).value
        self.alpha_4=self.declare_parameter('alpha_4',0.1).value


        self.period=self.declare_parameter('period',0.05).value
        self.odom_sub=self.create_subscription(Odometry,'odom',self.odom_callback,10)

        self.timer=self.create_timer(self.period,self.prediction_callback,)

        self.coordinates={} #dictionary with keys the id of the landmarks and the content should be (i hope) the vector [x,y,z] of the landmark
        # zip is usable only because the list have the same lenght otherwise it would stop to the shorter one, but we always have string of equal length
        for i,x,y,z in enumerate(zip(self.id,self.coordinates_x,self.coordinates_y,self.coordinates_z)):


            self.coordinates[i]=[float(x),float(y),float(z)]
          
          

        #initializes the filter variables using matrices
        self.mu=np.zeros(dim_x) #mean state value estimation
        self.sigma = np.eye(dim_x) #covariance of the estimation
        self.Mt=np.eye(dim_u) #noise of the command process
        
        

        self._I = np.eye(dim_x)  # identity matrix used for computations
        self.distances={}
           







     def odom_callback(self,msg):

            [self.x_act,self.y_act,self.z_act]=[msg.pose.pose.position.x,
                                 msg.pose.pose.position.y,
                                 msg.pose.pose.position.z ] # take the actual position of the robot in the map 

            self.quat=[msg.pose.pose.orientation.x,
                  msg.pose.pose.orientation.y,
                  msg.pose.pose.orientation.z,
                  msg.pose.pose.orientation.w ] # take the actual orientation of the robot in the map 

            [self.v,self.w]=[msg.twist.twist.linear.x,
                             msg.twist.twist.angular.z]

    

         

            
     """ def eval_Gt(self,u,prev_state,dt): #jacobian with respect to the state (x,y,theta_z or yaw)

         theta=prev_state[2,0] #state stands for (x,y,theta) from the odometry message

         v,w= u[0,0], u[1,0] #velocity command from the odometry message 

         Gt= np.array([[1, 0, (v/w)*(-math.cos(theta)+math.cos(theta+w*dt))],
                       [0,1,(v/w)*(-math.sin(theta)+math.sin(theta+w*dt))],
                       [0,0,1]])

         return Gt


     def eval_Vt(self,u,prev_state,dt): #jacobian with respect to the command

         theta=prev_state[2,0]
         
         v,w= u[0,0], u[1,0]

         Vt=np.array([[(-math.sin(theta)+math.sin(theta+w*dt))/w, (v/(w**2))*(math.sin(theta)-math.sin(theta+w*dt))+(v/w)*(math.cos(theta+w*dt)*dt)],
                      [(1/w)*(math.cos(theta)-math.cos(w*dt)), (-v/(w**2))*(math.cos(theta)-math.cos(theta+w*dt))+(v/w)*(math.sin(theta+w*dt)*dt)],
                      [0,dt]])

         return Vt

     def eval_Mt(self,u,alpha_1,alpha_2,alpha_3,alpha_4):

         v,w= u[0,0], u[1,0]

         Mt=np.array([[alpha_1*(v**2)+alpha_2*(w**2),0],
                      [0,alpha_3*(v**2)+alpha_4*(w**2)]]) #define the alpha 
         return Mt

     def eval_mu(self,u,prev_mu,dt): # evaluate the new predicted mean value

         theta=prev_mu[2,0]
         v,w_act= u[0,0], u[1,0]
         
         w= max(w_act,1e-9) 
              
         eval_mu= prev_mu + np.array([[(-v/w)*math.sin(theta)+(v/w)*math.sin(theta+w*dt)],
                                    [(v/w)*math.cos(theta)+(-v/w)*math.cos(theta+w*dt)],
                                    [w*dt]])
         return eval_mu """
         
    #  def distance(self,coordinates,mu):

    #       x,y,theta= mu[0],mu[1],mu[2]

    #       for id in coordinates:

    #           x_obs,y_obs,_= coordinates[id]

    #           lin_distance=math.hypot(x_obs-x,y_obs-y)

    #           angular_pos= math.atan2(y_obs-y,x_obs-x)-theta

    #           self.distances[id]=np.array([lin_distance],[angular_pos])

    #       return self.distances



     def prediction_callback(self):

         u=[self.v,self.w]
         _,_,yaw=euler_from_quaternion(self.quat)
         x=[self.x_act,self.y_act,yaw]
         dt= self.period
         alpha=[ self.alpha_1,self.alpha_2,self.alpha_3,self.alpha_4]# defined in the init
         dim_x=3

         if self.mu==np.zeros(dim_x):
              prev_state=np.array([x[0],x[1],x[2]])
              prev_sigma=np.eye(dim_x)
         else:
              prev_state=corrected_state
              prev_sigma=corrected_sigma

        
         pred_state=localization_utils.eval_mu(u,prev_state,dt)

         Gt=localization_utils.eval_Gt(u,prev_state,dt)
         Vt=localization_utils.eval_Vt(u,prev_state,dt)
         Mt=localization_utils.eval_Mt(u,alpha)
         Ht=localization_utils.Ht(pred_state,self.coordinates)

         


        

         












         


         





    





          










         



         
         
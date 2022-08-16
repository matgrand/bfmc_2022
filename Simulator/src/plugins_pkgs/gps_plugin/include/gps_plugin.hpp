#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>


#include "ros/ros.h"
#include "utils/localisation.h"

namespace gazebo
{
    namespace gps
    {   
        class GPS: public ModelPlugin
    	{
        private: 
            physics::ModelPtr                   m_model;
            ros::NodeHandlePtr		  nh;
            ros::Timer				  timer;

	    /** ROS INTEGRATION **/
            // A node use for ROS transport
            std::unique_ptr<ros::NodeHandle>    m_ros_node;

            // A ROS publisher
            ros::Publisher                      m_pubGPS;

            // The gps message
            utils::localisation            m_gps_pose;

            //deque of past gps messages
            std::deque<utils::localisation> m_gps_history;
            
            // last publish time
            double last_pub;
            double packet_loss_time;
            
            // simulate packet loss
            bool losing_pkg = false; 
            
            
        // Default constructor
        public: GPS();
        public: void Load(physics::ModelPtr, sdf::ElementPtr);
        public: void OnUpdate();        
        };
    };    
};

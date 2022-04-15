
#include "gps_plugin.hpp"

#define DEBUG true
#define GPS_STD 0.1 // [m]
#define PROB_OF_STARTING_LOSING_PKGS 0.05
#define MAX_TIME_PKG_LOSS 2.3 // [s]

namespace gazebo
{
    namespace gps
    {   
        GPS::GPS():ModelPlugin() {}
     		
        void GPS::Load(physics::ModelPtr model_ptr, sdf::ElementPtr sdf_ptr)
        {
          nh = boost::make_shared<ros::NodeHandle>();
          timer = nh->createTimer(ros::Duration(0.25), std::bind(&GPS::OnUpdate, this));

  			  // Save a pointer to the model for later use
  			  this->m_model = model_ptr;
  			
        	// Create topic name      	
        	std::string topic_name = "/automobile/localisation";
  	        
          // Initialize ros, if it has not already been initialized.
    			if (!ros::isInitialized())
    			{
      			  int argc = 0;
      			  char **argv = NULL;
      			  ros::init(argc, argv, "localisationNODEvirt", ros::init_options::NoSigintHandler);
    			}

          this->m_ros_node.reset(new ::ros::NodeHandle("/localisationNODEvirt"));

        	this->m_pubGPS = this->m_ros_node->advertise<utils::localisation>(topic_name, 2);
        	
          if(DEBUG)
          {
              std::cerr << "\n\n";
              ROS_INFO_STREAM("====================================================================");
              ROS_INFO_STREAM("[gps_plugin] attached to: " << this->m_model->GetName());
              ROS_INFO_STREAM("[gps_plugin] publish to:  "  << topic_name);
              ROS_INFO_STREAM("[gps_plugin] PROB_OF_STARTING_LOSING_PKGS:" << PROB_OF_STARTING_LOSING_PKGS );
              ROS_INFO_STREAM("[gps_plugin] MAX_TIME_PKG_LOSS:           " << MAX_TIME_PKG_LOSS);
              ROS_INFO_STREAM("[gps_plugin] GPS_STD:                     " << GPS_STD);
              ROS_INFO_STREAM("====================================================================\n\n");
          }
        }

        // Publish the updated values
        void GPS::OnUpdate()
        {
		float true_x = this->m_model->RelativePose().Pos().X();
		float true_y = abs(this->m_model->RelativePose().Pos().Y());
		float time_stamp = this->m_model->GetWorld()->SimTime().Float();
		this->m_gps_pose.timestamp  = time_stamp;
           	this->m_gps_pose.posA   = true_x + (rand() / (float)RAND_MAX * 2 * GPS_STD) - GPS_STD;
           	this->m_gps_pose.posB   = true_y + (rand() / (float)RAND_MAX * 2 * GPS_STD) - GPS_STD;
           	this->m_gps_pose.rotA   = this->m_model->RelativePose().Rot().Yaw();
           	this->m_gps_pose.rotB   = this->m_model->RelativePose().Rot().Yaw();
           	
           	///*
           	//package loss
           	if (this->losing_pkg) {
           		double time_diff = time_stamp - this->last_pub;
           		//ROS_INFO_STREAM("time_diff = " << time_diff);
           		//ROS_INFO_STREAM("packet_loss_time = " << this->packet_loss_time);
           		if ((time_diff > this->packet_loss_time) || (time_diff <= 0.0) ) { 
           			this->losing_pkg = false;
           		}
           	}
           	else {
           		bool start_losing_pkgs = (rand()/(float)RAND_MAX) < PROB_OF_STARTING_LOSING_PKGS; 
           		//ROS_INFO_STREAM("start_losing_pkgs = " << start_losing_pkgs);
           		if (start_losing_pkgs) {
           			this->packet_loss_time = MAX_TIME_PKG_LOSS * rand() / (float)RAND_MAX;
           			//ROS_INFO_STREAM("START->packet_loss_time = " << this->packet_loss_time);
           			this->losing_pkg = true;
           		}
           	}
           	//ROS_INFO_STREAM("losing_pkgs = " << this->losing_pkg);
           	//*/
           	
           	bool inside_no_signal_region = (true_x > 15.0*0.78) && (true_y > 15.0*0.4);
           	
           	bool can_publish = (!inside_no_signal_region) && (!this->losing_pkg);
           	//ROS_INFO_STREAM("can_publish = " << can_publish);
		if (can_publish) {
		      	this->m_pubGPS.publish(this->m_gps_pose);
		      	this->last_pub = time_stamp;
		}
        };      
    }; //namespace gps
    GZ_REGISTER_MODEL_PLUGIN(gps::GPS)
}; // namespace gazebo

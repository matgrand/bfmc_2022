/**
 * Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#include <app1.hpp>

serialHandlerNode_CPP::serialHandlerNode_CPP(SerialComManager& f_comManager, ResponseHandler& f_reponseHandler)
									:m_comManager(f_comManager),
									 m_responseHandler(f_reponseHandler)
{}

serialHandlerNode_CPP::~serialHandlerNode_CPP(){	
	m_responseHandler.detach(std::string("1"), m_callbackFncObj);
	m_responseHandler.detach(std::string("2"), m_callbackFncObj);
	m_responseHandler.detach(std::string("3"), m_callbackFncObj);
	m_responseHandler.detach(std::string("4"), m_callbackFncObj);
	m_responseHandler.detach(std::string("5"), m_callbackFncObj);
	m_responseHandler.detach(std::string("6"), m_callbackFncObj);
	
	delete m_callbackFncObj;
	// Close all threads
	m_comManager.closeAll();
		
}

void serialHandlerNode_CPP::print(std::string str){
	std::cout<<str<<std::endl;
}

void serialHandlerNode_CPP::init(ros::NodeHandle* nh) {				
	m_callbackFncObj = ResponseHandler::createCallbackFncPtr(&serialHandlerNode_CPP::print, this);
	
	// Attach the callback function to the following messages.
	m_responseHandler.attach(std::string("1"), m_callbackFncObj);
	m_responseHandler.attach(std::string("2"), m_callbackFncObj);
	m_responseHandler.attach(std::string("3"), m_callbackFncObj);
	m_responseHandler.attach(std::string("4"), m_callbackFncObj);
	m_responseHandler.attach(std::string("5"), m_callbackFncObj);
	m_responseHandler.attach(std::string("6"), m_callbackFncObj);
	
	Subscribing = nh->subscribe("/automobile/command", 1, &serialHandlerNode_CPP::funcCallback, this);
}

void serialHandlerNode_CPP::funcCallback(const std_msgs::String::ConstPtr& msg){
	rapidjson::Document doc;
	const char* c = msg->data.c_str();
	doc.Parse(c);
	if (doc.HasMember("action")) { 
		std::string command = doc["action"].GetString();
		if(command =="1") {
			if (doc.HasMember("speed")){ 
				m_comManager.sendSpeed(doc["speed"].GetFloat());
			}
			else { 
				ROS_INFO_STREAM("Invalid message"); 
			}
		} else if (command =="2") {
			if (doc.HasMember("steerAngle")){ 
				m_comManager.sendSteer(doc["steerAngle"].GetFloat());
			}
			else {
				ROS_INFO_STREAM("Invalid message"); 
			}
		} else if (command =="3") {
			if (doc.HasMember("steerAngle")){ 
				m_comManager.sendBrake(doc["steerAngle"].GetFloat());
			}
			else { 
				ROS_INFO_STREAM("Invalid message");
			}
		} else if (command =="4") {
			if (doc.HasMember("activate")){ 
				m_comManager.sendPidState(doc["activate"].GetBool());
			}
			else { 
				ROS_INFO_STREAM("Invalid message");
			}
		} else if (command =="5") {
			if (doc.HasMember("activate")){ 
				m_comManager.sendEncoderPublisher(doc["activate"].GetBool());
			}
			else { 
				ROS_INFO_STREAM("Invalid message");
			}
		} else if (command =="6") {
			if (doc.HasMember("kp"), doc.HasMember("ki"), doc.HasMember("kd"), doc.HasMember("tf")){ 
				m_comManager.sendPidParam(doc["kp"].GetFloat(), doc["ki"].GetFloat(), doc["kd"].GetFloat(), doc["tf"].GetFloat());
			}
			else { 
				ROS_INFO_STREAM("Invalid message");
			}
		} else {
			ROS_INFO_STREAM("Received UNKNOWN message");
		}
	} else {
		ROS_INFO_STREAM("Invalid message");
	}
   
   ROS_INFO("I heard: [%s]", msg->data.c_str());
};


int main(int argc, char **argv) {

	ros::init(argc, argv, "serialHandlerNODE_CPP"); 
	ros::NodeHandle nh;

	ResponseHandler  	l_responseHandler;
	SerialComManager 	l_communicationManager(l_responseHandler);	
	serialHandlerNode_CPP   commandObject(l_communicationManager, l_responseHandler);
	
	commandObject.init(&nh);
		
	ros::spin();
}

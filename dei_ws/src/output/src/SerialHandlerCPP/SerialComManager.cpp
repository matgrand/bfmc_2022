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


#include <SerialComManager.hpp>

/*
* This class handles the serial communication. It starts the responde handler thread and the writer handler thread. It forwards the actions to the writer. 
*/

SerialComManager::SerialComManager(BaseResponseHandler& f_responseHandler)
                                :SerialComManager(19200,"/dev/ttyACM0",f_responseHandler)
{
}

SerialComManager::SerialComManager( unsigned int                f_baudrate
                                    ,const std::string          f_dev
                                    ,BaseResponseHandler&        f_responseHandler)
                                    :m_responseHandler(f_responseHandler)
                                    ,m_io_service()
                                    ,m_io_serviceThread(NULL)
                                    ,m_responseHandlerThread(NULL)
	                                ,m_serialPort(m_io_service,f_baudrate,f_dev,m_responseHandler)
{
    m_io_serviceThread=new boost::thread(boost::bind(&boost::asio::io_service::run, &m_io_service));
    m_responseHandlerThread=new boost::thread(boost::bind(&BaseResponseHandler::_run, &m_responseHandler));
}

SerialComManager::~SerialComManager()
{
    delete m_io_serviceThread;
}

void SerialComManager::closeAll(){
    m_serialPort.close();
    m_responseHandler.deactive();
    m_responseHandlerThread->join();
    m_io_serviceThread->join();
}

// COMMENTARIILE


void SerialComManager::sendSpeed(float f_vel){
    std::string l_msg=message::speed(f_vel);
    m_serialPort.write(l_msg);
}

void SerialComManager::sendSteer(float f_ster){
    std::string l_msg=message::steer(f_ster);
    m_serialPort.write(l_msg);
}

void SerialComManager::sendBrake(float f_angle){
    std::string l_msg=message::brake(f_angle);
    m_serialPort.write(l_msg);
}

void SerialComManager::sendPidState(bool f_activate){
    std::string l_msg=message::pida(f_activate);
    m_serialPort.write(l_msg);
}

void SerialComManager::sendEncoderPublisher(bool f_activate){
    std::string l_msg=message::enpb(f_activate);
    m_serialPort.write(l_msg);
}

void SerialComManager::sendPidParam(float f_kp,float f_ki,float f_kd,float f_tf){
    std::string l_msg=message::pids(f_kp,f_ki,f_kd,f_tf);
    m_serialPort.write(l_msg);
}





//SerialComManager.hpp - Header file
//class declaration for serial communication handling

#ifndef SERIAL_COM_MANAGER_HPP
#define SERIAL_COM_MANAGER_HPP

#include <boost/asio.hpp>
#include <serialPortHandler.hpp>
#include <ResponseHandler.hpp>
#include <Message.hpp>

class SerialComManager{
    public:
        SerialComManager(unsigned int, const std::string,BaseResponseHandler&);
        SerialComManager(BaseResponseHandler&);
        virtual ~SerialComManager();
        // void start();
        void closeAll();

        void sendSpeed(float);
        void sendSteer(float);
        void sendBrake(float);
        void sendPidState(bool);
        void sendEncoderPublisher(bool);
        void sendPidParam(float,float,float,float);
        
    private:
        BaseResponseHandler&             m_responseHandler;
        boost::asio::io_service 		 m_io_service;
        boost::thread*                   m_io_serviceThread;
        boost::thread*                   m_responseHandlerThread;
        serialPortHandler 				m_serialPort;
};

#endif

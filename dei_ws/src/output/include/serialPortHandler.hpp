//serialPortHandler.h - Header file
//class declaration for serial sender/receiver

#ifndef SERIAL_PORT_HANDLER_HPP
#define SERIAL_PORT_HANDLER_HPP

#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/asio/serial_port.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <string>
#include <iostream>

#include <ResponseHandler.hpp>


using namespace std;

/// @brief Different ways a serial port may be flushed.
enum flush_type
{
  flush_receive = TCIFLUSH,
  flush_send = TCOFLUSH,
  flush_both = TCIOFLUSH
};

class serialPortHandler
{
public:
	serialPortHandler(boost::asio::io_service& io_service, unsigned int baud, const string& device,BaseResponseHandler& responseHandler);
	void write(std::string);
	void close();
	bool active();

private:
    // maximum amount of data to read in one operation
	static const int max_read_length = 512; 
	
	void read_start(void);
	void read_complete(const boost::system::error_code& error, size_t bytes_transferred);
	void do_writeString( std::string);
	void write_start(void);
	void write_complete(const boost::system::error_code& error);
	void do_close(const boost::system::error_code& error);
    void flush_serial_port(boost::asio::serial_port& serial_port, flush_type what, boost::system::error_code& error);

private:
	BaseResponseHandler&				m_responseHandler; // object response handler processes the response received
	boost::mutex 						m_writeMtx; // object avoids the parallel writing 

	bool active_; // remains true while this object is still operating
	boost::asio::io_service& io_service_; //the main IO service that runs this connection
	boost::asio::serial_port serialPort; //the serial port this instance is connected to
	// char read_msg_[max_read_length]; //data read from the socket
	deque<char> write_msgs_; //buffered write data
public:
	char read_msg_[max_read_length]; //data read from the socket
};

#endif